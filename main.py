import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import cv2
import os
import numpy as np
import torch
import shutil
from dust3r.utils.device import to_numpy
import open3d as o3d


def poisson_mesh_pipeline(pcd, output_mesh_path):
    """
    1. 포인트 클라우드 불러오기
    2. 노이즈 제거(Statistical Outlier Removal)
    3. 다운샘플링(Voxel Downsampling) [선택 사항]
    4. 노멀 계산(estimate_normals)
    5. Poisson Reconstruction (create_from_point_cloud_poisson)
    6. 저밀도 영역 제거
    7. 메쉬 저장
    """
    # 1. 포인트 클라우드 불러오기
    # print(f"[INFO] Loading point cloud from: {input_ply_path}")
    # pcd = o3d.io.read_point_cloud(input_ply_path)

    print("[INFO] Initial point cloud size:", np.asarray(pcd.points).shape[0])

    # 2. 노이즈 제거 (Statistical Outlier Removal)
    #   - 파라미터 nb_neighbors, std_ratio 조절해가며 실험 가능
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    pcd = pcd.select_by_index(ind)
    print("[INFO] After Statistical Outlier Removal:",
          np.asarray(pcd.points).shape[0], "points")

    # 3. (선택) 다운샘플링
    #   - 과도하게 밀집된 영역을 균등화해줄 수 있어 메쉬가 더 깔끔해질 때가 있음
    voxel_size = 0.0005  # 데이터 스케일에 맞게 조절
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"[INFO] After voxel downsampling({voxel_size}):",
          np.asarray(pcd.points).shape[0], "points")

    # # 4. 노멀 계산
    # #   - 메쉬 생성에 매우 중요한 부분
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #     radius=0.01,  # 반경도 데이터 스케일에 맞춰 조절
    #     max_nn=30
    # ))
    # pcd.orient_normals_consistent_tangent_plane(k=10)

    # 5. Poisson Reconstruction
    #   - depth, scale, linear_fit 파라미터 등을 상황에 맞춰 조절
    print("[INFO] Start Poisson Reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=11,        # 디테일 정도
        width=0,        # 0이면 자동 설정
        scale=1.1,      # 1보다 약간 크게 두면 경계선 부분 잡아내는데 도움이 됨
        linear_fit=False
    )
    print("[INFO] Mesh vertices before density filtering:",
          len(mesh.vertices))

    # 6. 저밀도 영역 제거
    #   - densities는 각 버텍스마다 할당된 밀도 값
    densities = np.asarray(densities)
    #   예시: 하위 10% 밀도 영역 제거
    density_threshold = np.quantile(densities, 0.1)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print("[INFO] Mesh vertices after density filtering:",
          len(mesh.vertices))

    # 노멀 재계산 (optional)
    mesh.compute_vertex_normals()

    # 7. 최종 메쉬 저장
    print(f"[INFO] Saving mesh to: {output_mesh_path}")
    o3d.io.write_triangle_mesh(output_mesh_path, mesh, print_progress=True)
    print("[INFO] Done.")

def save_cams(path, poses, focal, img_filenames, cc, orig_w, orig_h):
    poses = to_numpy(poses)
    cx, cy = cc
    
    scaleW = orig_w / 512
    scaleH = orig_h / 384
    focals = to_numpy(focal)
    
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
            f.write(str(focals[i][0] * scaleW) + ' 0 ' + str(orig_w / 2) + '\n')
            f.write('0 ' + str(focals[i][0] * scaleH)+ ' ' + str(orig_h /2 ) + '\n')
            f.write('0 0 1\n')

def save_depth(path, depth_map, img_name):
    original_max = np.max(depth_map)
    original_min = np.min(depth_map)
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    scaled_depth_map = (depth_map_normalized * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(path, "depth_" + img_name.split(".")[0] + ".png"), scaled_depth_map)
    with open(os.path.join(path,"depth_" + img_name.split(".")[0] + ".txt"), "w") as f:
        f.write(f"max: {original_max}\n")
        f.write(f"min: {original_min}\n")


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    dataset = 'dtu'

    exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
    train_img_idxs = [22, 25, 28]
    
    train_images = {
            "fern": ["image001", "image010", "image019"],
            "flower": ["image001", "image017", "image033"],
            "fortress": ["image001", "image021", "image041"],
            "horns": ["DJI_20200223_163017_967", "DJI_20200223_163053_863", "DJI_20200223_163225_243"],
            "leaves": ["image001", "image012", "image025"],
            "orchids": ["image001", "image012", "image023"],
            "room": ["DJI_20200226_143851_396", "DJI_20200226_143918_576", "DJI_20200226_143946_704"],
            "trex": ["DJI_20200223_163551_210", "DJI_20200223_163616_980", "DJI_20200223_163654_571"],
        }
    
    make_idxs = [i for i in range(49) if i not in exclude_idx]
    
    if dataset == 'dtu':
        source_path = "/home/airlabs/Dataset/DTU/dtu_4/"
    elif dataset == 'llff':
        source_path = "/home/airlabs/Dataset/LLFF/llff_8/"   
    
    output_root_path ="./data"
    os.makedirs(output_root_path, exist_ok=True)

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    # model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    
    for data in sorted(os.listdir(source_path)):
        print("=" * 100)
        print(f"[INFO] Processing {data} ...")
        dust3r_path = os.path.join(output_root_path, dataset, data, "dust3r")
        os.makedirs(dust3r_path, exist_ok=True)
        
        img_path = os.path.join(source_path, data)
        img_list = sorted(os.listdir(img_path))
        
        images_dest_path = os.path.join(dust3r_path, "images")
        os.makedirs(images_dest_path, exist_ok=True)

        # 이미지 파일들을 "images" 폴더로 복사
        for img_file in img_list:
            src_file = os.path.join(img_path, img_file)  # 원본 이미지 파일 경로
            dst_file = os.path.join(images_dest_path, img_file)  # 복사할 목적지 파일 경로
            
            # 파일 복사
            shutil.copy(src_file, dst_file)
        
        train_img = [os.path.join(img_path, im) for idx, im in enumerate(img_list)]

        if dataset == 'llff':
            train_img_idxs = [idx for idx, im in enumerate(img_list) if im.split(".")[0] in train_images[data]]
            first_img = train_img_idxs[0]
        elif dataset == 'dtu':
            first_img = "22"
        
        # load_images can take a list of images or a directory
        images = load_images(train_img, size=512)
        pairs = make_pairs(images, scene_graph=f'oneref-{first_img}', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)
        torch.cuda.empty_cache()
        # # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
        # scene = scene.clean_pointcloud()
        
        # preset_one_intrinsic = [472.2984] * len(train_img)
        # preset_full_intrinsic = [data for data in [preset_one_intrinsic] for _ in range(len(train_img_idxs))]
        
        # scene.preset_focal(preset_one_intrinsic)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        # retrieve useful values from scene:
        imgs_name = sorted(os.listdir(img_path))
        imgs = scene.imgs
        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        depths = scene.get_depthmaps()
        confidence_masks = scene.get_masks()
        intrinsics = scene.get_intrinsics()
        
        pcd = o3d.geometry.PointCloud()

        if dataset == 'dtu':
            filtered_pcd = [imgs[idx] for idx in train_img_idxs]
            filtered_confidence_masks = [confidence_masks[idx] for idx in train_img_idxs]
            filtered_pts3d = [pts3d[idx] for idx in train_img_idxs]
        elif dataset == 'llff':
            filtered_pcd = [imgs[idx] for idx in train_img_idxs]
            filtered_confidence_masks = [confidence_masks[idx] for idx in train_img_idxs]
            filtered_pts3d = [pts3d[idx] for idx in train_img_idxs]

        depth_path = os.path.join(dust3r_path, "depths")
        orig = os.path.join(dust3r_path, "originalSize_depths")
        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(orig, exist_ok=True)
        for idx in train_img_idxs:
            depth_map = np.array(depths[idx].cpu().detach())
            if dataset == 'llff':
                resized_depth = cv2.resize(depth_map, (504, 378), interpolation=cv2.INTER_AREA)
            elif dataset == 'dtu':
                resized_depth = cv2.resize(depth_map, (400, 300), interpolation=cv2.INTER_AREA) # Best for downscaling
            save_depth(path=orig, depth_map=depth_map, img_name=imgs_name[idx])
            save_depth(path=depth_path, depth_map=resized_depth, img_name=imgs_name[idx])

        # for idx, (im, conf_mask, pts) in enumerate(zip(imgs, confidence_masks, pts3d)):
        for idx, (im, conf_mask, pts) in enumerate(zip(filtered_pcd, filtered_confidence_masks, filtered_pts3d)):
            colors = im.reshape(-1, 3)
            points = pts.reshape(-1, 3).detach().cpu().numpy()
            mask = conf_mask.reshape(-1).detach().cpu().numpy()
            
            refined_points = points[mask]
            refined_colors = colors[mask]
            
            if idx == 0:
                total_points = refined_points
                total_colors = refined_colors
            else :
                total_points = np.vstack([total_points, refined_points])
                total_colors = np.vstack([total_colors, refined_colors])
        
        pcd.points = o3d.utility.Vector3dVector(total_points)
        pcd.colors = o3d.utility.Vector3dVector(total_colors)
        
        pcd_single = o3d.geometry.PointCloud()
        
        mask_single = confidence_masks[train_img_idxs[0]].reshape(-1).detach().cpu().numpy()
        points_single = pts3d[train_img_idxs[0]].reshape(-1, 3).detach().cpu().numpy()
        colors_single = imgs[train_img_idxs[0]].reshape(-1, 3)
        
        pcd_single.points = o3d.utility.Vector3dVector(points_single[mask_single])
        pcd_single.colors = o3d.utility.Vector3dVector(colors_single[mask_single])
        pcd_single.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        
        # Compute normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=10)

        output_dir = os.path.join(dust3r_path, "ply")
        os.makedirs(output_dir, exist_ok=True)
        
        pcd = pcd.voxel_down_sample(voxel_size=0.0005)
        o3d.io.write_point_cloud(os.path.join(output_dir, "points3D.ply"), pcd)
        print("Saved points3D.ply to", output_dir)
        print(f"Point Cloud num: {np.array(pcd.points).shape[0]}")        
        
        poisson_mesh_pipeline(pcd_single, os.path.join(output_dir, "poisson_mesh_depth_10.ply"))
        
        if dataset == 'llff':
            save_cams(dust3r_path, poses, focals, imgs_name, [512/2, 384/2], 504, 378)
        elif dataset == 'dtu':
            save_cams(dust3r_path, poses, focals, imgs_name, [512/2, 384/2], 400, 300)
        
        print(f"[INFO] Processing {data} Done.")
        print("=" * 100  + "\n")
        
        # with open("dtu_82.pkl", "wb") as f:
        #     print(f"File object: {f}")  # 파일 객체 확인 
        #     pickle.dump({
        #         "poses": poses,
        #         "depth": scene.get_depthmaps(),
        #         "intrinsic": intrinsics,
        #         "im": imgs,
        #         "confidence_mask": confidence_masks,
        #         "pts": pts3d,
        #         "points": np.array(pcd.points),
        #         "colors": np.array(pcd.colors)
        #     }, f)