import numpy as np
import torch
from plyfile import PlyData
from pykeops.torch import LazyTensor
from scene.gaussian_model import GaussianModel  # GaussianModel 클래스 불러오기
from tqdm import tqdm

def load_mesh(ply_path, scale):
    """
    .ply 파일을 읽어 mesh 데이터를 불러옵니다.
    vertices, colors, normals를 리턴합니다.
    """
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex']

    vertices = np.stack(
        [vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=1
    ).astype(np.float32) * scale

    colors = np.stack(
        [vertex_data['red'], vertex_data['green'], vertex_data['blue']], axis=1
    ).astype(np.float32) / 255.0

    # ply에 존재한다면 법선 정보까지 로드
    normals = np.stack(
        [vertex_data['nx'], vertex_data['ny'], vertex_data['nz']], axis=1
    ).astype(np.float32)

    return vertices, colors, normals

def compute_soft_min_D_in_batches(
    gaussian_centers, mesh_vertices, mesh_colors, predicted_colors, tau, batch_size
):
    """
    각 Gaussian center와 모든 mesh 정점 사이의 유클리디안 거리로부터
    soft-min distance( -tau * logsumexp(-D/tau) )를 구하고,
    메시의 색상을 거리 기반 가중치로 평균낸 뒤,
    그 평균색상과 가우시안 예측 색상 간의 L2 색상 오차 합을 계산합니다.

    Returns:
      soft_min_D: shape (N,)
      color_dist_sum: float, 모든 가우시안에 대한 color dist의 합
    """
    soft_min_list = []
    color_dist_sum = 0.0
    N = gaussian_centers.shape[0]

    # 배치 단위로 나눠서 처리
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # LazyTensor로 변환
        x_i_batch = LazyTensor(gaussian_centers[start:end, None, :])  # (B, 1, 3)
        y_j = LazyTensor(mesh_vertices[None, :, :])                   # (1, M, 3)

        # 거리 행렬 D_ij = ||x_i - y_j||^2
        D_ij_batch = ((x_i_batch - y_j) ** 2).sum(-1)  # (B, M)

        # soft-min distance = -tau * logsumexp(-D / tau)
        lse_batch = (-D_ij_batch / tau).logsumexp(dim=1)   # (B,)
        soft_min_batch = -tau * lse_batch                  # (B,)
        soft_min_list.append(soft_min_batch)

        # color distance 계산:
        #   weights = softmax(-D_ij / tau)
        #   weighted_color = sum(weights * mesh_colors)
        weights = (-D_ij_batch / tau - lse_batch[:, None]).exp()  # (B, M)
        weighted_color = weights @ mesh_colors                    # (B, 3)
        pred_colors_batch = predicted_colors[start:end, :]        # (B, 3)

        diff = pred_colors_batch - weighted_color
        norms = diff.norm(dim=1)                  # (B,)
        color_dist_sum += norms.sum().item()      # 배치 합을 누적

    soft_min_D = torch.cat(soft_min_list, dim=0)  # (N,)
    return soft_min_D, color_dist_sum

def compute_nn_and_sign_in_batches(
    gaussian_centers, mesh_vertices, mesh_normals,
    dot_threshold=0.0,
    batch_size=131072
):
    """
    각 가우시안 중심에 대해 '가장 가까운 정점'을 찾고,
    해당 정점의 normal과 (가우시안 중심 - 정점) 벡터의 dot product로
    내부/외부를 판별하여 sign(±1)을 반환합니다.

    Returns:
      outside_mask: dot_product > dot_threshold 인지 (bool 텐서)
      signs: +1 (outside) 또는 -1 (inside) (float 텐서)
    """
    N = gaussian_centers.shape[0]
    outside_mask_list = []
    sign_list = []

    # 텐서 변환
    gaussian_centers_t = torch.as_tensor(gaussian_centers, dtype=torch.float32)
    mesh_vertices_t = torch.as_tensor(mesh_vertices, dtype=torch.float32)
    mesh_normals_t = torch.as_tensor(mesh_normals, dtype=torch.float32)

    # 배치 처리
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # (B, 3)
        centers_batch = gaussian_centers_t[start:end, :]
        x_i_batch = LazyTensor(centers_batch[:, None, :])    # (B, 1, 3)
        y_j = LazyTensor(mesh_vertices_t[None, :, :])        # (1, M, 3)

        # 거리 계산
        D_ij = ((x_i_batch - y_j) ** 2).sum(-1)  # (B, M)

        # 가장 가까운 정점 인덱스 (B,)
        idx_min_batch = D_ij.argmin(dim=1)

        # 최근접 정점, 법선 추출
        nn_vertices_batch = mesh_vertices_t[idx_min_batch, :]  # (B, 3)
        nn_normals_batch = mesh_normals_t[idx_min_batch, :]    # (B, 3)

        # direction: (center - nearest_vertex)
        nn_vertices_batch = nn_vertices_batch.squeeze(1) 
        dir_batch = centers_batch - nn_vertices_batch          # (B, 3)

        # dot product
        nn_normals_batch = nn_normals_batch.squeeze(1) 
        dot_val = (dir_batch * nn_normals_batch).sum(dim=1)    # (B,)

        # dot_threshold 대비 양수면 바깥, 음수면 안쪽이라고 가정
        outside_mask_batch = (dot_val > dot_threshold)
        sign_batch = torch.where(outside_mask_batch, 1.0, -1.0)

        outside_mask_list.append(outside_mask_batch)
        sign_list.append(sign_batch)

    outside_mask = torch.cat(outside_mask_list, dim=0)  # (N,)
    signs = torch.cat(sign_list, dim=0)                 # (N,)

    return outside_mask, signs

def get_mesh_point_dist_with_normal_consistency(
    mesh_vertices, mesh_colors, mesh_normals,
    gaussians: GaussianModel,
    is_final,
    dot_threshold=0.0,
    sigma_factor=2,
    tau=1e-4,
    batch_size=131072
):
    """
    1) GaussianModel에서 포인트와 색상 불러오기
    2) soft-min distance 계산
    3) 거리 기반 threshold로 dist_mask 생성
    4) '가장 가까운 정점 1개'의 normal과 dot product를 사용해 내부/외부 판별
    5) SDF sign 부여 후 pruning (is_final=True인 경우)
    """
    # 1) GaussianModel에서 포인트/색상 불러오기
    gaussian_centers_tensor = gaussians.get_xyz  # (N, 3)
    gaussian_centers = gaussian_centers_tensor.to(dtype=torch.float32).contiguous()

    # 예측 색상 (N, 3)이라고 가정
    predicted_colors = gaussians._features_dc.squeeze(1).to(dtype=torch.float32).contiguous()

    # 2) soft-min distance 계산
    soft_min_D, color_dist_sum = compute_soft_min_D_in_batches(
        gaussian_centers, mesh_vertices, mesh_colors,
        predicted_colors, tau, batch_size
    )
    print("calc min D done")

    # 3) 거리 기준 threshold 설정
    mean_val = soft_min_D.mean()
    std_val = soft_min_D.std()
    dist_thres = mean_val + sigma_factor * std_val
    print(f"Distance Threshold = {dist_thres.item():.4f}")

    dist_mask = (soft_min_D > dist_thres).squeeze()

    # 4) 가장 가까운 정점의 normal과 dot product -> 내부/외부 판별
    outside_mask, signs = compute_nn_and_sign_in_batches(
        gaussian_centers, mesh_vertices, mesh_normals,
        dot_threshold=dot_threshold,
        batch_size=batch_size
    )

    # soft_min_D에 sign(±1)을 곱해 SDF 비슷하게 사용
    device = soft_min_D.device
    signs = signs.to(device)
    soft_min_D = soft_min_D.squeeze()
    sdf_values = soft_min_D * signs
    sdf_loss = sdf_values.abs().sum()
    print(f"sdf loss: {sdf_loss.item():.4f}")

    sum_of_distances = soft_min_D.sum()
    color_dist_sum = torch.tensor(color_dist_sum).to(device)

    print("dist loss:", sum_of_distances.item())
    print("color loss:", color_dist_sum.item())

    # 최종 Pruning
    if is_final:
        # dist_thres도 넘고 + 바깥쪽이라고 판정(outside_mask)된 점들을 제거
        final_mask = dist_mask & outside_mask.to(device)
        gaussians.prune_points(final_mask)
        print(f"Num points to prune = {final_mask.sum().item()}")

    return sum_of_distances, color_dist_sum, sdf_loss
