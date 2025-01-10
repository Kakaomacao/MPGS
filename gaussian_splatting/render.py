#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
import numpy as np
from os import makedirs
import open3d as o3d
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, save_mask
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.sh_utils import eval_sh

def C2I(intrinsic, points_camera, colors_camera):
    # 카메라 좌표계의 점들을 numpy 배열로 변환
    # points_camera_np = np.asarray(points_camera.points)

    # 투영 과정 (camera space -> image space)
    projected_points = np.dot(intrinsic, points_camera[:, :3].T).T
    
    # 동차 좌표계에서 (x, y) 평면 좌표를 얻기 위해 z 좌표로 나누기
    projected_points[:, :2] = projected_points[:, :2] / projected_points[:, 2:3]
    
    u, v = projected_points[:, 0], projected_points[:, 1]    

    # 이미지 범위를 벗어난 좌표는 클리핑
    u = np.clip(u, 0, intrinsic[0][2] * 2 - 1)
    v = np.clip(v, 0, intrinsic[1][2] * 2 - 1)
    
    # Stack and return the 2D image coordinates
    image_coords = np.vstack((u, v)).T
    colors_projected = colors_camera
    
    return image_coords, colors_projected

def project_gs(gaussians, view):
    # load GS
    pcd = o3d.geometry.PointCloud()
    
    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
    dir_pp = (gaussians.get_xyz)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    
    point = o3d.utility.Vector3dVector(gaussians.get_xyz.detach().cpu().numpy())
    color = o3d.utility.Vector3dVector(colors_precomp.detach().cpu().numpy())
    
    pcd.points = point
    pcd.colors = color
    
    # project GS
    W2C = view.world_view_transform.transpose(0, 1)
    pcd.transform(W2C.cpu().numpy())
    
    h, w = view.image_height, view.image_width
    intrinsic = view
    
    # Project points to 2D
    points_3d = np.asarray(pcd.points)
    points_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # Nx4 (homogeneous coordinates)
    projected_h = np.dot(C2I.detach().cpu(), points_h.T).T  # Nx4 -> Nx3
    projected_2d = projected_h[:, :2] / projected_h[:, 2:3]  # Normalize by z

    # Clamp points within image bounds
    u = np.clip(projected_2d[:, 0], 0, w - 1).astype(np.int32)
    v = np.clip(projected_2d[:, 1], 0, h - 1).astype(np.int32)
    
    # Generate white background
    white_background = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Draw points on the white background
    for idx, (x, y) in enumerate(zip(u, v)):
        if 0 <= x < w and 0 <= y < h:  # Ensure points are in valid range
            color = np.asarray(pcd.colors)[idx] * 255  # Convert color to [0, 255]
            cv2.circle(
                white_background,
                (x, y),
                radius=2,
                color=(int(color[0]), int(color[1]), int(color[2])),
                thickness=-1
            )
            
    # Show the white background with points
    cv2.imshow("Projected Gaussian Points", white_background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([pcd, axes])
    

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, masked_psnr=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    if masked_psnr:
        masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask")
        makedirs(masks_path, exist_ok=True)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        
        # TODO: Test Cam Reprojection Error로 수정
        # 1. GS Projection
        # 2. Cam 기준 GS Projection
        # 3. L1 loss 계산
        # 4. Cam Pose 조정
        # project_gs(gaussians, view)
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if masked_psnr:
            save_mask(os.path.join(masks_path, '{0:05d}'.format(idx) + ".png"), view.fg_mask)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            masked_psnr = (dataset.dataset == 'DTU')
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, masked_psnr=masked_psnr)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    args.dtu_mask_path = '/home/airlabs/Dataset/DTU/submission_data/idrmasks' # 필수
    args.resolution = 1
    args.dataset = "LLFF"
    # args.source_path = "/home/airlabs/Dataset/LLFF/llff_8/fern"
    # args.model_path = "/home/airlabs/SuGaR/gaussian_splatting/output/LLFF/fern_novel_mask_copy"
    
    # args.dataset = "DTU"
    # args.source_path = f"/home/airlabs/Dataset/DTU/dtu_4/scan8"
    # args.model_path = "/home/airlabs/SuGaR/gaussian_splatting/output/DTU/scan8"
    # args.sh_degree = 3
    # args.images = "distorted"
    # args.white_background = False
    # args.data_device = "cuda"
    args.eval = True
    # args.novelTrain = True
    args.input_views = 3

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)