import os
import numpy as np
import torch
from pykeops.torch import LazyTensor
import open3d as o3d
from plyfile import PlyData, PlyElement
import argparse


##############################################################################
# 1) 메시 로드 (Open3D) : 좌표, 노말 추출
##############################################################################

def load_mesh_open3d(mesh_path, scale=1.0):
    """
    Open3D를 통해 .ply(또는 .obj 등) 메시 파일을 읽어와
    vertices와 vertex_normals를 np.array로 반환.
    만약 파일에 노말이 없다면, compute_vertex_normals()로 계산.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    return vertices, normals


##############################################################################
# 2) 포인트 클라우드 로드 (plyfile) : 모든 property 보존
##############################################################################

def load_pointcloud_with_all_properties(ply_path, scale=1.0):
    """
    plyfile을 통해 .ply 포인트 클라우드를 읽어
    - 구조화 배열(structured array)로 모든 property를 보존
    - (N,3) 좌표 배열만 별도로 추출
    반환합니다.

    예) property가
       x, y, z, red, green, blue, intensity 등
       모든 필드가 구조화 배열에 들어있고,
       points_xyz에는 (x,y,z)만 들어갑니다.
    """
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex']  # PlyElement
    structured_array = np.asarray(vertex_data)  # structured array

    # x, y, z만 모아서 (N, 3) float 배열
    # scale 적용
    points_xyz = np.column_stack((structured_array['x'],
                                  structured_array['y'],
                                  structured_array['z'])).astype(np.float32)
    points_xyz *= scale

    return structured_array, points_xyz


##############################################################################
# 3) Soft-min 거리 계산 (색상/기타 불필요)
##############################################################################

def compute_soft_min_D_in_batches(point_centers, mesh_vertices, tau=1e-4, batch_size=131072):
    """
    각 point_centers와 모든 mesh_vertices 간의 유클리디안 거리로부터
    soft-min distance( -tau * logsumexp(-D/tau) )를 계산해 반환합니다.

    Returns:
      soft_min_D: shape (N,) in torch.Tensor
    """
    N = point_centers.shape[0]
    soft_min_list = []

    # 텐서 변환
    point_centers_t = torch.as_tensor(point_centers, dtype=torch.float32)
    mesh_vertices_t = torch.as_tensor(mesh_vertices, dtype=torch.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        x_i_batch = LazyTensor(point_centers_t[start:end, None, :])  # (B, 1, 3)
        y_j       = LazyTensor(mesh_vertices_t[None, :, :])          # (1, M, 3)

        D_ij_batch = ((x_i_batch - y_j) ** 2).sum(-1)  # (B, M)

        # soft-min distance = -tau * logsumexp(-D / tau)
        lse_batch = (-D_ij_batch / tau).logsumexp(dim=1)  # (B,)
        soft_min_batch = -tau * lse_batch                 # (B,)

        soft_min_list.append(soft_min_batch)

    soft_min_D = torch.cat(soft_min_list, dim=0)  # (N,)
    return soft_min_D


##############################################################################
# 4) 최근접 정점 기반 내부/외부 판별
##############################################################################

def compute_nn_and_sign_in_batches(point_centers, mesh_vertices, mesh_normals,
                                   dot_threshold=0.0,
                                   batch_size=131072):
    """
    각 point_center에 대해 가장 가까운 mesh 정점/노말을 찾아,
    dot product > dot_threshold => outside, <= => inside
    Returns: outside_mask(bool), signs(+1/-1)
    """
    point_centers_t = torch.as_tensor(point_centers, dtype=torch.float32)
    mesh_vertices_t = torch.as_tensor(mesh_vertices, dtype=torch.float32)
    mesh_normals_t  = torch.as_tensor(mesh_normals,  dtype=torch.float32)

    N = point_centers_t.shape[0]
    outside_mask_list = []
    sign_list = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        centers_batch = point_centers_t[start:end, :]
        x_i_batch = LazyTensor(centers_batch[:, None, :])  # (B,1,3)
        y_j       = LazyTensor(mesh_vertices_t[None, :, :])# (1,M,3)

        D_ij = ((x_i_batch - y_j) ** 2).sum(-1)  # (B,M)
        idx_min_batch = D_ij.argmin(dim=1)       # (B,)

        nn_vertices_batch = mesh_vertices_t[idx_min_batch, :]
        nn_normals_batch  = mesh_normals_t[idx_min_batch, :]

        nn_vertices_batch = nn_vertices_batch.squeeze(1)
        nn_normals_batch = nn_normals_batch.squeeze(1)

        dir_batch = centers_batch - nn_vertices_batch
        dot_val = (dir_batch * nn_normals_batch).sum(dim=1)  # (B,)

        outside_mask_batch = (dot_val > dot_threshold)
        sign_batch = torch.where(outside_mask_batch, 1.0, -1.0)

        outside_mask_list.append(outside_mask_batch)
        sign_list.append(sign_batch)

    outside_mask = torch.cat(outside_mask_list, dim=0)
    signs = torch.cat(sign_list, dim=0)

    return outside_mask, signs


##############################################################################
# 5) 거리+노말 기반 Pruning & 전체 property 보존
##############################################################################

def prune_pointcloud_by_mesh(
    mesh_vertices, mesh_normals,
    pcd_structured_array,   # 모든 property가 있는 구조화 배열
    pcd_points_xyz,         # (N,3) 좌표만
    dot_threshold=0.0,
    sigma_factor=2,
    tau=1e-4,
    batch_size=131072,
    is_final=True
):
    """
    1) soft-min distance (pcd_points_xyz vs mesh_vertices)
    2) dist_thres = mean + sigma_factor * std
    3) dot product > dot_threshold => outside
    4) (dist_thres 초과 & outside) => prune
    5) 최종적으로 pcd_structured_array에서 해당 점 제거
    """
    # (1) soft-min distance
    soft_min_D = compute_soft_min_D_in_batches(
        pcd_points_xyz, mesh_vertices, tau=tau, batch_size=batch_size
    )
    print("[INFO] Soft-min distance computed.")

    # (2) 거리 기반 threshold
    mean_val = soft_min_D.mean()
    std_val  = soft_min_D.std()
    dist_thres = mean_val + sigma_factor * std_val
    print(f"[INFO] Distance threshold = {dist_thres.item():.4f}")

    dist_mask = (soft_min_D > dist_thres)

    # (3) 내부/외부 판별
    outside_mask, signs = compute_nn_and_sign_in_batches(
        pcd_points_xyz, mesh_vertices, mesh_normals,
        dot_threshold=dot_threshold,
        batch_size=batch_size
    )

    # (soft_min_D * signs) => sdf 유사값
    soft_min_D = soft_min_D.squeeze(1)
    sdf_values = soft_min_D * signs
    sdf_loss = sdf_values.abs().sum()
    print(f"[INFO] sdf loss = {sdf_loss.item():.4f}")

    sum_of_distances = soft_min_D.sum()
    print("[INFO] dist loss =", sum_of_distances.item())

    # (4) Pruning
    #     "제거할" 점 = dist_mask & outside_mask
    dist_mask = dist_mask.squeeze(1)
    final_mask = dist_mask & outside_mask

    if is_final:
        print(f"[INFO] Num points to prune = {final_mask.sum().item()}")
    else:
        print("[INFO] is_final=False, no actual pruning mask applied.")

    # (5) 최종 구조화 배열에서 prune
    #     final_mask = True => 제거 대상
    pruned_structured = pcd_structured_array[~final_mask.cpu().numpy()]  
    return sum_of_distances, sdf_loss, pruned_structured, final_mask


##############################################################################
# 6) PLY 저장 함수 : 구조화 배열 전체 저장
##############################################################################

def save_ply_structured_array(structured_array, output_path):
    """
    'structured_array'를 그대로 PlyElement로 만들어
    모든 property를 포함한 상태로 저장합니다.
    """
    el = PlyElement.describe(structured_array, 'vertex')
    PlyData([el], text=False).write(output_path)


##############################################################################
# 7) 메인 (argparse) - 인자를 주지 않아도 내부에서 기본값 사용
##############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, default=None,
                        help="Mesh .ply path (Open3D로 로드)")
    parser.add_argument("--pcd_path", type=str, default=None,
                        help="PointCloud .ply path (plyfile로 모든 property 로드)")
    parser.add_argument("--out_path", type=str, default=None,
                        help="Pruned PCD .ply 저장 경로")
    parser.add_argument("--scale", type=float, default=None,
                        help="좌표 스케일 (기본값=1.0)")
    parser.add_argument("--dot_threshold", type=float, default=None,
                        help="노말 dot product 임계값 (기본=0.0)")
    parser.add_argument("--sigma_factor", type=float, default=None,
                        help="soft-min 거리 임계값 factor (기본=2)")
    parser.add_argument("--tau", type=float, default=None,
                        help="soft-min 파라미터 (기본=1e-4)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="pyKeOps batch_size (기본=131072)")
    args = parser.parse_args()

    # 사용자 미입력시 내부 기본값
    if args.mesh_path is None:
        args.mesh_path = "data/DTU/scan8/dust3r/ply/poisson_mesh_depth_10.ply"
    if args.pcd_path is None:
        args.pcd_path = "output/DTU/scan8/point_cloud/iteration_1/point_cloud.ply"
    if args.out_path is None:
        args.out_path = "output/DTU/scan8/point_cloud/iteration_1/point_cloud_pruned.ply"
    if args.scale is None:
        args.scale = 1.0
    if args.dot_threshold is None:
        args.dot_threshold = 0.0
    if args.sigma_factor is None:
        args.sigma_factor = 2.0
    if args.tau is None:
        args.tau = 1e-4
    if args.batch_size is None:
        args.batch_size = 131072

    print("===== ARGS =====")
    print(f"mesh_path     = {args.mesh_path}")
    print(f"pcd_path      = {args.pcd_path}")
    print(f"out_path      = {args.out_path}")
    print(f"scale         = {args.scale}")
    print(f"dot_threshold = {args.dot_threshold}")
    print(f"sigma_factor  = {args.sigma_factor}")
    print(f"tau           = {args.tau}")
    print(f"batch_size    = {args.batch_size}")
    print("================")

    # (A) 메시 로드 (Open3D)
    mesh_vertices, mesh_normals = load_mesh_open3d(args.mesh_path, scale=args.scale)
    print(f"[INFO] Loaded mesh: {mesh_vertices.shape[0]} vertices")

    # (B) PCD 로드 (plyfile) => 모든 property + (N,3) 좌표
    pcd_structured, pcd_points_xyz = load_pointcloud_with_all_properties(args.pcd_path, scale=args.scale)
    print(f"[INFO] Loaded point cloud: {pcd_points_xyz.shape[0]} points (all properties)")

    # (C) 거리+노말 기반 Pruning (최종 mask 적용)
    dist_loss, sdf_loss, pruned_structured, final_mask = prune_pointcloud_by_mesh(
        mesh_vertices=mesh_vertices,
        mesh_normals=mesh_normals,
        pcd_structured_array=pcd_structured,
        pcd_points_xyz=pcd_points_xyz,
        dot_threshold=args.dot_threshold,
        sigma_factor=args.sigma_factor,
        tau=args.tau,
        batch_size=args.batch_size,
        is_final=True
    )
    print(f"[INFO] After pruning => {pruned_structured.shape[0]} points remain")

    # (D) 구조화 배열 전체를 ply로 저장 (모든 property 보존)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    save_ply_structured_array(pruned_structured, args.out_path)
    print(f"[INFO] Saved pruned PCD with all properties => {args.out_path}")


if __name__ == "__main__":
    main()
