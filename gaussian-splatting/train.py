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

import matplotlib
matplotlib.use('TkAgg')

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

from gaussian_grad_switcher import get_grad_switched_gaussians

# def show_plot(image):
#     if isinstance(image, torch.Tensor):
#         image= image.squeeze()    
#         if image.dim() == 3:
#             if image.shape[0] == 3:
#                 image = image.detach().cpu().numpy().transpose(1, 2, 0)
#             else :
#                 image = image.detach().cpu().numpy()
#     if isinstance(image, np.ndarray):
#         if image.shape[0] == 3:
#             image = image.transpose(1, 2, 0)
    
#     plt.imshow(image)
#     plt.show()
    

def create_camera_actor(scale=0.3, color=(1.0, 0.0, 0.0)):
    # Define camera points (normalized)
    CAM_POINTS = np.array([
        [0, 0, 0],        # Camera origin
        [-1, -1, 2],      # Bottom-left of the frustum
        [1, -1, 2],       # Bottom-right of the frustum
        [1, 1, 2],        # Top-right of the frustum
        [-1, 1, 2]        # Top-left of the frustum
    ]) * scale

    # Define connections between points
    CAM_LINES = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from origin to frustum corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # Lines connecting the frustum edges
    ])

    # Create LineSet object
    camera_actor = o3d.geometry.LineSet()
    camera_actor.points = o3d.utility.Vector3dVector(CAM_POINTS)
    camera_actor.lines = o3d.utility.Vector2iVector(CAM_LINES)
    camera_actor.paint_uniform_color(color)

    return camera_actor


def vis_gs(pc, cam=None, save=False, mask=None):
    from utils.sh_utils import eval_sh
    pcd = o3d.geometry.PointCloud()
    
    if mask is not None:
        shs_view = pc.get_features[mask].transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz)[mask]
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        point = o3d.utility.Vector3dVector(pc.get_xyz[mask].detach().cpu().numpy())
        color = o3d.utility.Vector3dVector(colors_precomp.detach().cpu().numpy())
        
        pcd.points = point
        pcd.colors = color
    
    else:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        point = o3d.utility.Vector3dVector(pc.get_xyz.detach().cpu().numpy())
        color = o3d.utility.Vector3dVector(colors_precomp.detach().cpu().numpy())
        
        pcd.points = point
        pcd.colors = color
    
    if cam is not None:
        total_list = [pcd]
        if isinstance(cam, list):
            for c in cam:
                cam_actor = create_camera_actor()
                ref_pose = np.eye(4)
                R, T = c.R, c.T
                ref_pose[:3, :3] = R.transpose()
                ref_pose[:3, 3] = T

                ref_pose = np.linalg.inv(ref_pose)
                cam_actor.transform(ref_pose)
                # cam_actor.rotate(R)
                # cam_actor.translate(T)
                total_list.append(cam_actor)
        else :
            cam_actor = create_camera_actor(True)
            c = cam
            ref_pose = np.eye(4)
            R, T = c.R, c.T
            ref_pose[:3, :3] = R
            ref_pose[:3, 3] = T
            
            ref_pose = np.linalg.inv(ref_pose)
            cam_actor.transform(ref_pose)
            # cam_actor.rotate(R)
            # cam_actor.translate(T)
            total_list.append(cam_actor)
        return o3d.visualization.draw_geometries(total_list)
    
    if save:
        o3d.io.write_point_cloud("test.ply", pcd)
        
    o3d.visualization.draw_geometries([pcd])


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg, rasterizer = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, is_return_rasterizer=True)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        if viewpoint_cam.nv_mask is not None:
            mask = torch.from_numpy(viewpoint_cam.nv_mask).to("cuda").bool()
            mask_3d = mask.unsqueeze(0)
            mask_3d = mask_3d.repeat(3, 1, 1)
            Ll1 = l1_loss(image, gt_image, mask_3d)
        else:
            Ll1 = l1_loss(image, gt_image)
            
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Isotropic loss from MonoGS
        # scaling = gaussians.get_scaling
        # isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
        # loss += isotropic_loss.mean()
        
        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 :
            invDepth = render_pkg["depth"]
            if viewpoint_cam.nv_mask is not None:
                invDepth = invDepth
                depth = torch.from_numpy(viewpoint_cam.depth).cuda()
            else:
                depth = torch.from_numpy(viewpoint_cam.depth).cuda()

            depth = torch.clamp(depth, min=1e-8)
            depth_reciprocal = torch.reciprocal(depth).cuda()
            Ll1depth_pure = torch.abs(invDepth  - depth_reciprocal).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()
        
        if viewpoint_cam.nv_mask is not None:
            grad_mask = get_grad_switched_gaussians(gaussians, viewpoint_cam, rasterizer)

            def mask_gradients(grad):
                grad[grad_mask] = 0
                return grad

            # Gradient 수정에 Hook 사용
            gaussians._xyz.register_hook(mask_gradients)
            gaussians._features_dc.register_hook(mask_gradients)
            gaussians._features_rest.register_hook(mask_gradients)
            gaussians._scaling.register_hook(mask_gradients)
            gaussians._rotation.register_hook(mask_gradients)
            gaussians._opacity.register_hook(mask_gradients)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 6_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 6_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    target = "scan38"
    args.dataset = "DTU"
    args.eval = True
    args.novelTrain = True
    
    if args.dataset == "LLFF":
        args.source_path = f"./data/llff/{target}"
    elif args.dataset == "DTU":
        args.source_path = f"./data/dtu/{target}"
    args.model_path = f"output/{args.dataset}/{target}_pcd_mesh_novel"
    args.iteration = 20_000

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
