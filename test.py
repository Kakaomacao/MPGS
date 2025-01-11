import os
import torch
import numpy as np
import PIL.Image
import open3d as o3d

from mast3r.model import AsymmetricMASt3R

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.demo import get_3D_model_from_scene
from dust3r.utils.device import to_numpy

from read_write_model import read_model, qvec2rotmat

# 데이터 설정
target_data = "scan82"
# source_path = f"/home/airlabs/Dataset/LLFF/llff_8/{target_data}"
source_path = f'/home/airlabs/Dataset/DTU/dtu_4/{target_data}'

def main():
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    
    colmap_path = os.path.join(source_path, "sparse", "0")
    full_input_path = os.path.join(source_path, 'images')
    input3_path = os.path.join(source_path, '3')
    output_path = os.path.join(source_path, "dust3r_test")
    
    colmap_infos = get_camera_info(colmap_path, os.listdir(input3_path) ,input_format=".bin")
    known_poses, known_focals = extract_camera_info(colmap_infos)
    
    images = load_images(full_input_path, size=512)
    pairs = make_pairs(images, scene_graph='oneref-22', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, optimize_pp=True)
    # scene.preset_pose(known_poses=known_poses)
    # scene.preset_focal(known_focals=known_focals)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    
    imgs0 = scene.imgs
    poses0 = scene.get_im_poses()
    focals0 = scene.get_focals()
    av_focal0 = (sum(focals0) / len(focals0)).item() 
    imgs_name = sorted([f for f in os.listdir(full_input_path) if f.lower().endswith(('.jpg', '.png', '.JPG', '.PNG'))])
    cc = [imgs0[0].shape[1] / 2.0, imgs0[0].shape[0] / 2.0]
    
    orig_images = PIL.Image.open(os.path.join(full_input_path, imgs_name[0]))
    W, H = orig_images.size
    
    # CAM 정보 저장
    save_cams(output_path, poses0, av_focal0, imgs_name, cc, orig_w=W, orig_h=H)
    print(f"Saved camera parameters to {output_path}/cams")
    
    # 전체 이미지 PCD 저장
    ply_path_all = get_3D_model_from_scene(scene=scene, outdir=os.path.join(output_path, "ply"), silent=False)
    
    # 3장에 대한 PCD 저장
    imgs_name_all = sorted([f for f in os.listdir(full_input_path) if f.lower().endswith(('.jpg', '.png', '.JPG', '.PNG'))])
    target_images = sorted(os.listdir(input3_path))  # 3장 이미지 이름 로드
    target_indices = [imgs_name_all.index(img) for img in target_images]  # 전체 이미지 중 해당 3장의 인덱스 추출
    ply_path_3 = get_3D_model_from_scene(scene=scene, outdir=os.path.join(output_path, "ply"), silent=False, selected_images=target_indices)

    # # pcd Outlier Removal
    # pcd = o3d.io.read_point_cloud(ply_path)
    # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    # pcd_radius = pcd.select_by_index(ind)    
    # cl, ind = pcd_radius.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # pcd_statistical = pcd_radius.select_by_index(ind)
    # o3d.visualization.draw_geometries([pcd_statistical])

# =================================================================================================

def save_cams(path, poses, focal, img_filenames, cc, orig_w, orig_h):
    poses = to_numpy(poses)
    cx, cy = cc
    
    scaleW = orig_w / 512
    scaleH = orig_h / 384
    
    for i in range(len(img_filenames)):
        pose = poses[i]
        img_filename = img_filenames[i]
        txt_filename = img_filename.split('.')[0] + '_cam.txt'
        folder_path = path + '/cams/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(folder_path + txt_filename, 'w') as f:
            f.write('extrinsic\n')
            for row in pose:
                line = ' '.join([str(val) for val in row])
                f.write(line + '\n')
            f.write('\nintrinsic\n')
            f.write(str(focal * scaleW) + ' 0 ' + str(orig_w / 2) + '\n')
            f.write('0 ' + str(focal * scaleH)+ ' ' + str(orig_h /2 ) + '\n')
            f.write('0 0 1\n')

def get_camera_info(colmap_output_path, image_names, input_format=".bin"):
    """
    COLMAP 결과에서 카메라 정보를 읽어 반환하는 함수.
    
    Args:
        colmap_output_path (str): COLMAP 결과 폴더 경로.
        input_format (str): COLMAP 파일 형식 ('.bin' 또는 '.txt').

    Returns:
        dict: 카메라 정보를 담은 딕셔너리.
    """
    # COLMAP 결과 읽기
    cameras, images, points3D = read_model(path=colmap_output_path, ext=input_format)
    
    filtered_images = {img_id: img for img_id, img in images.items() if img.name in image_names}
    # 카메라 정보를 JSON-friendly dict로 변환
    cameras_dict = []
    for img_id, img in filtered_images.items():
        # Rotation Matrix (R) 계산
        R = qvec2rotmat(img.qvec).tolist()
        T = img.tvec.tolist()
        
        width = cameras[img_id].width
        height = cameras[img_id].height
        fx = cameras[img_id].params[0]
        fy = cameras[img_id].params[1]
        cx = cameras[img_id].params[2]
        cy = cameras[img_id].params[3]
        
        cameras_dict.append({
            "img_name": img.name,
            "width": width,
            "height": height,
            "R": R,
            "T": T,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        })
    
    return cameras_dict

def extract_camera_info(cameras_dict):
    """
    카메라 정보 딕셔너리에서 pose, focal, principal point 정보를 추출하는 함수.
    
    Args:
        cameras_dict (dict): 카메라 정보를 담은 딕셔너리.

    Returns:
        tuple: pose, focal, principal point 정보를 담은 튜플.
    """
    poses = torch.empty((0, 4, 4))
    focals = []
    
    for cam in cameras_dict:
        extrinsic = torch.eye(4)
        extrinsic[:3, :3] = torch.tensor(cam["R"])
        extrinsic[:3, 3] = torch.tensor(cam["T"])
        extrinsic = extrinsic.unsqueeze(0)
        poses = torch.cat((poses, extrinsic), dim=0)
        
        scale = 512 / cam["width"]
        focals.append(cam["fx"] * scale)
        
    return poses, focals

# =================================================================================================

if __name__ == '__main__':
    main()