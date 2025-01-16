import numpy as np
import torch
from plyfile import PlyData
from pykeops.torch import LazyTensor
from scene.gaussian_model import GaussianModel  # GaussianModel 클래스 불러오기
from tqdm import tqdm

def load_mesh(ply_path, scale):
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex']

    vertices = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=1).astype(np.float32) * scale
    colors = np.stack([vertex_data['red'], vertex_data['green'], vertex_data['blue']], axis=1).astype(np.float32) / 255.0

    # 법선 정보 추출 (존재할 경우)
    normals = np.stack([vertex_data['nx'], vertex_data['ny'], vertex_data['nz']], axis=1).astype(np.float32)
    
    return vertices, colors, normals

def compute_soft_min_D_in_batches(gaussian_centers, mesh_vertices, mesh_colors, predicted_colors, tau, batch_size):
    soft_min_list = []
    color_dist_sum = 0.0
    N = gaussian_centers.shape[0]
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        # LazyTensor 생성
        x_i_batch = LazyTensor(gaussian_centers[start:end, None, :])
        y_j = LazyTensor(mesh_vertices[None, :, :])

        D_ij_batch = ((x_i_batch - y_j) ** 2).sum(-1)
        lse_batch = (-D_ij_batch / tau).logsumexp(dim=1)
        soft_min_list.append(-tau * lse_batch)

        weights = (-D_ij_batch / tau - lse_batch[:, None]).exp()
        weighted_color = weights @ mesh_colors
        pred_colors_batch = predicted_colors[start:end, :] 
        
        diff = pred_colors_batch - weighted_color 
        norms = diff.norm(dim=1)                  
        color_dist_sum += norms.sum().item()  # 배치 내 합산

    soft_min_D = torch.cat(soft_min_list, dim=0)
    return soft_min_D, color_dist_sum

def compute_knn_and_average_dot_in_batches(gaussian_centers, mesh_vertices, mesh_normals, K=5, batch_size=131072):
    """
    각 가우시안 중심점에 대해 K개의 가까운 메시 정점을 찾아,
    해당 정점들의 법선과의 dot product 평균을 계산하여 반환합니다.
    """
    N = gaussian_centers.shape[0]
    avg_dot_list = []

    # 입력 데이터를 텐서로 변환
    gaussian_centers = torch.as_tensor(gaussian_centers, dtype=torch.float32)
    mesh_vertices_t = torch.as_tensor(mesh_vertices, dtype=torch.float32)
    mesh_normals_t = torch.as_tensor(mesh_normals, dtype=torch.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        
        # LazyTensor 생성
        x_i_batch = LazyTensor(gaussian_centers[start:end, None, :])  # (B, 1, 3)
        y_j = LazyTensor(mesh_vertices_t[None, :, :])                 # (1, M, 3)

        # 배치별 거리 계산
        D_ij = ((x_i_batch - y_j) ** 2).sum(-1)  # shape: (B, M)
        
        # 각 점에 대해 K개의 최근접 이웃 인덱스 계산
        idx_k_batch = D_ij.argKmin(K=K, dim=1)  # shape: (B, K)

        # 해당 배치의 가우시안 점들
        centers_batch = gaussian_centers[start:end, :]  # shape: (B, 3)
        
        # K개의 이웃 정점과 법선 추출
        nn_vertices_batch = mesh_vertices_t[idx_k_batch.view(-1), :].view(idx_k_batch.shape[0], K, -1)  # (B, K, 3)
        nn_normals_batch = mesh_normals_t[idx_k_batch.view(-1), :].view(idx_k_batch.shape[0], K, -1)    # (B, K, 3)

        # 가우시안 점과 이웃 정점 사이의 방향 벡터 계산
        centers_expanded = centers_batch.unsqueeze(1)            # (B, 1, 3)
        dir_batch = centers_expanded - nn_vertices_batch         # (B, K, 3)

        # 각 이웃 법선과의 dot product 계산
        dot_batch = (dir_batch * nn_normals_batch).sum(dim=2)    # (B, K)

        # K개 이웃에 대한 평균 dot 계산
        avg_dot = dot_batch.mean(dim=1)                          # (B,)

        avg_dot_list.append(avg_dot.cpu())

    avg_dot_values = torch.cat(avg_dot_list, dim=0)
    return avg_dot_values

def get_mesh_point_dist_with_normal_consistency(
    mesh_vertices, mesh_colors, mesh_normals,
    gaussians: GaussianModel, is_final,
    dot_threshold=0.0,
    sigma_factor=2.5,
    tau=1e-4,
    batch_size=131072,
    K=5
):
    # 1) GaussianModel에서 포인트/색상 불러오기
    gaussian_centers_tensor = gaussians.get_xyz  
    gaussian_centers = gaussian_centers_tensor.to(dtype=torch.float32).contiguous()
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

    # 4) 다수의 근접 이웃 평균 법선 일치성 계산
    avg_dot_values = compute_knn_and_average_dot_in_batches(
        gaussian_centers, mesh_vertices, mesh_normals, K=K, batch_size=batch_size
    )

    # 외부 여부 및 SDF sign 결정
    # avg_dot_values는 CPU에 있으므로 GPU로 옮기거나, 비교 연산을 CPU에서 진행
    device = soft_min_D.device
    avg_dot_values = avg_dot_values.to(device)
    outside_mask = (avg_dot_values > dot_threshold).to(device)
    signs = torch.where(avg_dot_values > dot_threshold, 1.0, -1.0).to(device)

    # 근사 Signed Distance 계산
    soft_min_D = soft_min_D.squeeze()
    sdf_values = soft_min_D * signs
    sdf_loss = sdf_values.abs().sum()
    print(f"sdf loss: {sdf_loss.item():.4f}")

    if is_final:
        final_mask = dist_mask & outside_mask
        gaussians.prune_points(final_mask)

    sum_of_distances = soft_min_D.sum()
    color_dist_sum = torch.tensor(color_dist_sum).to(device)

    print("dist loss:", sum_of_distances.item())
    print("color loss:", color_dist_sum.item())
    if is_final:
        print(f"Num points to prune = {final_mask.sum().item()}")

    return sum_of_distances, color_dist_sum, sdf_loss
