from diff_gaussian_rasterization import GaussianRasterizer
import torch
import numpy as np
from scene import GaussianModel
from scene.cameras import Camera

def get_grad_switched_gaussians(gaussians: GaussianModel, viewpoint_cam: Camera, rasterizer: GaussianRasterizer):
    """
    gaussians : GaussianModel
        - gaussians.get_xyz -> shape (N, 3) on CUDA
    viewpoint_cam : Camera
        - viewpoint_cam.nv_mask -> (H, W) bool mask (True/False) 또는 numpy.ndarray
        - viewpoint_cam.full_proj_transform -> (4, 4) transform matrix 
        - viewpoint_cam.image_width, viewpoint_cam.image_height
        - 필요한 경우 기타 카메라 속성들
    rasterizer : GaussianRasterizer
        - 이미 설정된 rasterizer 인스턴스

    Returns:
        refined_visible : torch.BoolTensor (N,)
            - NV 마스크 조건을 만족하며 현재 카메라 시점에 보이는 가우시안 표시
    """
    device = "cuda"

    # nv_mask가 없는 경우, 모든 가우시안이 보인다고 가정
    if viewpoint_cam.nv_mask is None:
        return torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device=device)

    # 외부에서 제공된 rasterizer를 사용하여 가시성 마스크 계산
    visible_mask = rasterizer.markVisible(gaussians.get_xyz)

    # 가시한 가우시안들의 인덱스 추출
    visible_indices = torch.nonzero(visible_mask).squeeze()
    if visible_indices.numel() == 0:
        return visible_mask

    # visible_mask로 필터링된 가우시안들의 위치
    xyz_visible = gaussians.get_xyz[visible_indices]
    ones_col = torch.ones((xyz_visible.size(0), 1), device=device)
    points_4d = torch.cat([xyz_visible, ones_col], dim=1)
    projected = points_4d @ viewpoint_cam.full_proj_transform

    x_ndc = projected[:, 0] / projected[:, 3]
    y_ndc = projected[:, 1] / projected[:, 3]
    z_ndc = projected[:, 2] / projected[:, 3]
    valid_z = (z_ndc > 0) & (z_ndc < 1)

    W = viewpoint_cam.image_width
    H = viewpoint_cam.image_height
    u = ((x_ndc * 0.5) + 0.5) * W
    v = ((-y_ndc * 0.5) + 0.5) * H
    u = u.long()
    v = v.long()
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid_all = valid_z & in_bounds

    # nv_mask를 torch.Tensor로 변환
    if isinstance(viewpoint_cam.nv_mask, np.ndarray):
        mask_tensor = torch.from_numpy(viewpoint_cam.nv_mask).to(device)
    else:
        mask_tensor = viewpoint_cam.nv_mask.to(device)

    final_mask = torch.zeros_like(valid_all, dtype=torch.bool, device=device)
    final_mask[valid_all] = mask_tensor[v[valid_all], u[valid_all]]

    # markVisible로 얻은 가시성 마스크를 nv_mask 조건으로 정제
    refined_visible = torch.zeros_like(visible_mask, dtype=torch.bool, device=device)
    refined_visible[visible_indices] = final_mask

    return refined_visible