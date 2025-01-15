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

import os
import sys
import cv2
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from dataclasses import dataclass

@dataclass
class CameraInfo:
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    
    image: np.array
    image_path: str
    image_name: str
    
    depth: np.array
    depth_path: str
    depth_bounds: tuple
    
    width: int
    height: int
    # is_test: bool
    
    fg_mask: np.array = None # for DTU evaluation
    nv_mask: np.array = None # for Novel Train View

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    # is_nerf_synthetic: bool

def fetchPly_scale(path, scale):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T * scale
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list, load_fg_mask=False, dtu_mask_path=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        # read dtu foregroud mask
        dtu_mask = None
        if load_fg_mask:
            scene_name = image_path.split('/')[-3]
            idx = int(image_name.split('_')[1]) - 1
            dtu_mask_file = os.path.join(dtu_mask_path, scene_name, f'{idx:03d}.png')
            if not os.path.exists(dtu_mask_file):
                dtu_mask_file = os.path.join(dtu_mask_path, scene_name, f'{idx:03d}.png')
            if os.path.exists(dtu_mask_file):
                dtu_mask = np.array(Image.open(dtu_mask_file), dtype=np.float32)[:, :, :3] / 255.
                dtu_mask = (dtu_mask == 1)[:,:,0]

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):
    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

def readJsonCamerasDUST3R(json_path, images_folder, scale=50):
    with open(json_path, "r") as f:
        cameras_data = json.load(f)
    
    cam_infos = []
    
    for cam in cameras_data:
        uid=cam["id"]
        img_name=cam["img_name"] + ".png"
        c2w = np.array(cam["extrinsic"])
        c2w = c2w.transpose()
        K = np.array(cam["intrinsic"])
        
        image_path = os.path.join(images_folder, "images", img_name)
        image = Image.open(image_path)
        W1, H1 = image.size
        
        depth = None
        depth_path = os.path.join(images_folder, "depths")
        depth_name = "depth_" + cam["img_name"] + ".png"
        if os.path.exists(os.path.join(depth_path, depth_name)):           
            depth_bounds_txt = os.path.join(depth_path, "depth_" + cam["img_name"] + ".txt")
            if os.path.exists(depth_bounds_txt):
                with open(depth_bounds_txt, 'r') as f:
                    lines = f.readlines()
                max = float(lines[0].split(":")[1].strip())
                min = float(lines[1].split(":")[1].strip())
                depth_bounds = (max, min)
            else:
                raise Exception("Error message: no depth bounds file exits")
            depth_file = os.path.join(depth_path, depth_name)
            depth = (cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE))
            depth = depth / 255.0
            depth = depth * (depth_bounds[0] - depth_bounds[1]) + depth_bounds[1]

        w2c = np.linalg.inv(c2w)
        # w2c = c2w
        R = np.transpose(w2c[:3,:3])
        # R = w2c[:3,:3]
        T = w2c[:3, 3] * scale

        W2, H2 = K[0][2]*2, K[1][2]*2  
        FovX = focal2fov(K[0,0],W2)
        FovY = focal2fov(K[1,1],H2)

        nv_mask = None
        mask_name = "mask_" + cam["img_name"] + ".png"
        if os.path.exists(os.path.join(images_folder, mask_name)):
            mask_file = os.path.join(images_folder, mask_name)
            nv_mask = (cv2.imread(mask_file, 0))
            nv_mask = (nv_mask != 0)
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, 
                              image=image, image_path=image_path, image_name=img_name,
                              depth=depth, depth_path=depth_path, depth_bounds=depth_bounds,
                              width=W1, height=H1, nv_mask=nv_mask)
        cam_infos.append(cam_info)
        
    return cam_infos

def readCamerasFromDUST3R(img_path, cam_path, depth_path, load_fg_mask=False, dtu_mask_path=None, scale = 50):
    cam_infos = []
    image_files = sorted(os.listdir(img_path))
    cam_files = sorted(os.listdir(cam_path))    
    idx = 0    
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        idx += 1
        cam_file = f"{image_name}_cam.txt"
        if cam_file in cam_files:
            image_path = os.path.join(img_path, image_file)
            camera_path = os.path.join(cam_path, cam_file)
        else :
            raise Exception("Error message: no cam file exits matched with{image_file}")
        
        image = Image.open(image_path)
        W1, H1 = image.size

        depth = None
        depth_bounds = None
        depth_name = f"depth_{image_name}.png"
        if os.path.exists(os.path.join(depth_path, depth_name)):           
            depth_bounds_txt = os.path.join(depth_path, f"depth_{image_name}.txt")
            if os.path.exists(depth_bounds_txt):
                with open(depth_bounds_txt, 'r') as f:
                    lines = f.readlines()
                max = float(lines[0].split(":")[1].strip())
                min = float(lines[1].split(":")[1].strip())
                depth_bounds = (max, min)
            else:
                raise Exception("Error message: no depth bounds file exits")
            depth_file = os.path.join(depth_path, depth_name)
            depth = (cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE))
            depth = depth / 255.0
            depth = depth * (depth_bounds[0] - depth_bounds[1]) + depth_bounds[1]

        with open(camera_path, 'r') as file:
            lines = file.readlines()

        c2w = []
        for i in range(1,5):
            line = lines[i].strip().split()
            row = [float(val) for val in line]
            c2w.append(row)
        c2w = np.array(c2w)
        w2c = np.linalg.inv(c2w) # Dust3r은 원래 w2c라서 GS는 c2w로 바꿔줘야함 근데 지금 Novel 만들때 바꿔서 저장 중이니까 그냥 사용(GS는 c2w를 받음) (변수명 잘못 표기)
        # w2c = c2w
        R = np.transpose(w2c[:3,:3])  
        # R = w2c[:3,:3]
        T = w2c[:3, 3] * scale

        K = []
        for i in range(7, 10):
            line = lines[i].strip().split()
            row = [float(val) for val in line]
            K.append(row)
        K = np.array(K)
        W2, H2 = K[0][2]*2, K[1][2]*2  
        FovX = focal2fov(K[0,0],W2)
        FovY = focal2fov(K[1,1],H2)
        
        # DTU foreground mask Load
        dtu_mask = None
        if load_fg_mask:
            scene_name = image_path.split('/')[-3]
            idx = int(image_name.split('_')[1]) - 1
            dtu_mask_file = os.path.join(dtu_mask_path, scene_name, f'{idx:03d}.png')
            if not os.path.exists(dtu_mask_file):
                dtu_mask_file = os.path.join(dtu_mask_path, scene_name, f'{idx:03d}.png')
            if os.path.exists(dtu_mask_file):
                dtu_mask = np.array(Image.open(dtu_mask_file), dtype=np.float32)[:, :, :3] / 255.
                dtu_mask = (dtu_mask == 1)[:,:,0]
        
        nv_mask = None
        
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, 
                                    image=image, image_path=image_path, image_name=image_name,
                                    depth=depth, depth_path=depth_path, depth_bounds=depth_bounds,
                                    width=W1, height=H1, fg_mask=dtu_mask, nv_mask=nv_mask))
            
    return cam_infos

def readCamerasFromDUST3Rtest(img_path, cam_path, white_background, extension=".jpg", scale = 50, train_image = None, dep_path = None, nor_path = None):
    #for zju competition, not know the img
    cam_infos = []
    image_files = sorted(os.listdir(img_path))
    cam_files = sorted(os.listdir(cam_path))
    idx = 0
    for cam_file in cam_files:
        image_name = os.path.splitext(cam_file)[0][:-4]
        idx += 1
        image_file = image_name + extension
        image_path = os.path.join(img_path, image_file)
        camera_path = os.path.join(cam_path, cam_file)

        if image_name in image_files:
            image = Image.open(image_path)
            W1, H1 = image.size
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        else:
            image = train_image
            W1, H1 = image.width, image.height
       
        depth = None
        normal = None

        with open(camera_path, 'r') as file:
            lines = file.readlines()

        c2w = []
        for i in range(1,5):
            line = lines[i].strip().split()
            row = [float(val) for val in line]
            c2w.append(row)
        c2w = np.array(c2w)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  
        T = w2c[:3, 3] * scale

        K = []
        for i in range(7, 10):
            line = lines[i].strip().split()
            row = [float(val) for val in line]
            K.append(row)
        K = np.array(K)
        W2, H2 = K[0][2]*2, K[1][2]*2  
        FovX = focal2fov(K[0,0],W2)
        FovY = focal2fov(K[1,1],H2)
        intrinsic = np.array([fov2focal(FovX, W1), fov2focal(FovY, H1), W1/2, H1/2])
        intrinsic = np.array([[fov2focal(FovX, W1), 0, W1/2],[0, fov2focal(FovY, H1), H1/2],[0,0,1]])
        extrinsic = np.eye(4)
        extrinsic[:3,:3] = w2c[:3,:3]
        extrinsic[:3,3] = T
        
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, normal=normal,
                    image_path=image_path, image_name=image_name, width=W1, height=H1, intrinsics=intrinsic, extrinsics=extrinsic))
            
    return cam_infos

def readDUST3RInfo(path, images, depths, eval, llffhold=8, dataset='DTU', input_n=3, dtu_mask_path=None, novelTrainView=False, extension=".jpg", white_background=False): 
    scale = 1  # dust3r scale is too small, 3dgs SIBR viewer cannot see, so we scale 100
    
    images = "images" if images == None else images
    depths = "depths" if depths == None else depths
    dust_dir = os.path.join(path, "dust3r")
    load_fg_mask = True if dataset=='DTU' else False
    
    if os.path.exists(os.path.join(dust_dir, "cams")) and os.path.exists(os.path.join(dust_dir, images)) and os.path.exists(os.path.join(dust_dir, depths)):
        cams_folder = os.path.join(dust_dir, "cams")
        images_folder = os.path.join(dust_dir, images)
        depths_folder = os.path.join(dust_dir, depths)
    else:
        raise Exception("Error message: no cams folder exits")    
    all_cam_infos = readCamerasFromDUST3R(images_folder, cams_folder, depths_folder, load_fg_mask, dtu_mask_path, scale)
    cam_infos = sorted(all_cam_infos.copy(), key = lambda x : x.image_name)
    
    print('Dataset: ', dataset)
    # dataset split
    if eval:
        if dataset == 'DTU':
            print('Eval DTU Dataset!!!')
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
            test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
            train_cam_infos = [cam_infos[i] for i in train_idx[:input_n]]
            print(f"Train image name : {[c.image_name for c in train_cam_infos]}")
            test_cam_infos = [cam_infos[i] for i in test_idx]
        elif dataset == 'LLFF':
            print('Eval LLFF Dataset!!!')
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            if input_n >= 1:
                idx_sub = np.linspace(0, len(train_cam_infos) - 1, input_n)
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
                print(f"Train image name : {[c.image_name for c in train_cam_infos]}")
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
    
    if novelTrainView:
        print('NovelTrainView!!!')
        novel_path = os.path.join(path, "novel_views")
        json_file = os.path.join(novel_path, "cams_novelTrainView.json")
        new_cam_infos = readJsonCamerasDUST3R(json_file, images_folder=novel_path, scale=scale)
        train_cam_infos += new_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    if novelTrainView:
        ply_path = os.path.join(path, "novel_views","sampled_points_300000.ply")
    else:
        ply_path = os.path.join(dust_dir, "ply", "points3D.ply")
        

    try:
        pcd = fetchPly_scale(ply_path, scale)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "DUST3R": readDUST3RInfo,
}