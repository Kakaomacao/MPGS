from scene import Scene, GaussianModel
from scene.cameras import Camera
import numpy as np
import matplotlib.pyplot as plt
import torch


def get_grad_switched_gaussians(gaussians: GaussianModel, viewpoint_cam: Camera):
    """
    gaussians : GaussianModel
        - gaussians.get_xyz -> shape (N, 3) on CUDA
    viewpoint_cam : Camera
        - viewpoint_cam.nv_mask -> (H, W) bool mask (True/False) (또는 numpy.ndarray)
        - viewpoint_cam.full_proj_transform -> (4, 4) transform matrix 
          (projection_matrix x world_view_transform),  row-vector 기준
        - viewpoint_cam.image_width, viewpoint_cam.image_height
    Returns:
        selected_mask : torch.BoolTensor (N,)
            - True인 가우시안만 nv_mask 상에서 보이는(투영된 위치가 True) 점
    """
    device = gaussians.get_xyz.device

    # nv_mask가 없는 경우, 전부 True 처리
    if viewpoint_cam.nv_mask is None:
        return torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device=device)

    # 만약 viewpoint_cam.nv_mask가 numpy라면 -> torch 텐서로 변환
    if isinstance(viewpoint_cam.nv_mask, np.ndarray):
        mask_tensor = torch.from_numpy(viewpoint_cam.nv_mask).to(device)
    else:
        # 이미 torch 텐서라고 가정
        mask_tensor = viewpoint_cam.nv_mask.to(device)

    # 1) 가우시안들의 xyz 가져오기
    xyz = gaussians.get_xyz  # shape (N, 3)

    # 2) homogeneous 좌표로 확장
    ones_col = torch.ones((xyz.size(0), 1), device=device)
    points_4d = torch.cat([xyz, ones_col], dim=1)  # (N, 4)

    # 3) full_proj_transform (world->view->proj) 적용
    #    row-vector 곱이므로: p' = p * M_full
    projected = points_4d @ viewpoint_cam.full_proj_transform

    # 4) 여기서는 z를 [0,1]로 가정(DirectX/Vulkan 스타일)
    x_ndc = projected[:, 0] / projected[:, 3]
    y_ndc = projected[:, 1] / projected[:, 3]
    z_ndc = projected[:, 2] / projected[:, 3]

    # 5) z_ndc가 0 < z < 1인 점만 "카메라 앞"이라고 가정
    valid_z = (z_ndc > 0) & (z_ndc < 1)

    # 6) NDC(-1..1) -> 픽셀 (u, v)
    W = viewpoint_cam.image_width
    H = viewpoint_cam.image_height
    # x_ndc = -1 => u=0, x_ndc=1 => u=W
    u = (x_ndc * 0.5 + 0.5) * W
    # y_ndc = -1 => v=H, y_ndc=1 => v=0  (상단 원점)
    v = (-y_ndc * 0.5 + 0.5) * H

    u = u.long()
    v = v.long()

    # 7) 이미지 범위 안에 있는지
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid_all = valid_z & in_bounds

    # 8) nv_mask에서 True인지 확인
    #    mask_tensor shape: (H, W), v:행, u:열
    final_mask = torch.zeros_like(valid_all, dtype=torch.bool)
    final_mask[valid_all] = mask_tensor[v[valid_all], u[valid_all]]

    return final_mask
