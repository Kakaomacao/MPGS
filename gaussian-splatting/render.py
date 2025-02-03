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
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, save_mask
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.sh_utils import eval_sh
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, separate_sh, masked_psnr=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    if masked_psnr:
        masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask")
        makedirs(masks_path, exist_ok=True)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        if masked_psnr:
            save_mask(os.path.join(masks_path, '{0:05d}'.format(idx) + ".png"), view.fg_mask)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, separate_sh)

        if not skip_test:
            masked_psnr = (dataset.dataset == 'DTU')
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, separate_sh, masked_psnr=masked_psnr)

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
    # print("Rendering " + args.model_path)

    args.dtu_mask_path = '/home/lsw/Dataset/DTU/submission_data/idrmasks' # 필수
    args.resolution = 1
    # args.dataset = "LLFF"
    # args.source_path = "/home/airlabs/Dataset/LLFF/llff_8/leaves"
    # args.model_path = "/home/airlabs/MPGS/output/LLFF/leaves_vanila"
    
    args.dataset = "DTU"
    # args.source_path = "./data/DTU/scan38"
    # args.model_path = "/home/airlabs/MPGS/output/236cfc08-6"
    args.sh_degree = 3
    args.images = "images"
    args.depths = "depths"
    args.data_device = "cuda"
    args.eval = True
    args.novelTrain = True
    args.input_views = 3
    args.white_background = False

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)