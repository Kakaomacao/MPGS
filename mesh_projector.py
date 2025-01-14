import torch
import math
import numpy as np
import open3d as o3d
import json
import os
from tqdm import tqdm
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import copy


def main():
    # 설정!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dataset_type = "DTU"    # 데이터셋 타입 (DTU 또는 LLFF)
    framework = "dust3r"    # 사용할 프레임워크 (dust3r 또는 colmap)
    n = 300_000             # Poisson Disk Sampling 할 점의 수
    
    # 소스 설정
    if dataset_type == "DTU":
        source_root = f"/home/airlabs/Dataset/DTU/dtu_4" 
    elif dataset_type == "LLFF":
        source_root = f"/home/airlabs/Dataset/LLFF/llff_8"

    # 각 children 폴더에 대해 처리
    for target_data in tqdm(os.listdir(source_root), desc=f"Processing {dataset_type} with {framework}", position=0, leave=True):
        target_dir_path = os.path.join(source_root, target_data)
        if not os.path.isdir(target_dir_path):
            continue
        
        # 해당 폴더에서 .jpg 또는 .JPG로 끝나는 파일 이름 추출
        desired_numbers = {"022", "025", "028"}
        
        desired_images = {
            "fern": {"001", "010", "019"},
            "flower": {"001", "017", "033"},
            "fortress": {"001", "021", "041"},
            "horns": {"DJI_20200223_163017_967", "DJI_20200223_163053_863", "DJI_20200223_163225_243"},
            "leaves": ["001", "012", "025"],
            "orchids": {"001", "012", "023"},
            "room": {"DJI_20200226_143851_396", "DJI_20200226_143918_576", "DJI_20200226_143946_704"},
            "trex": {"DJI_20200223_163551_210", "DJI_20200223_163616_980", "DJI_20200223_163654_571"},
        }

        # train_image_names 설정
        if dataset_type == "DTU":
            # DTU: rect_{desired_numbers}_3_r5000 형식
            train_image_names = [f"rect_{num}_3_r5000" for num in desired_numbers]
        elif dataset_type == "LLFF":
            # LLFF: desired_images에서 매칭된 이름 사용
            if target_data in desired_images:
                train_image_names = [f"{name}" for name in desired_images[target_data]]
            else:
                train_image_names = []
                print(f"Warning: No desired images specified for {target_data}")
        
        # 경로 설정 (원하는 경로로 변경)
        target_dir = os.path.join(target_dir_path, "dust3r_test")
        input3_mesh_dir = os.path.join(target_dir, "ply", "poisson_mesh_depth_10.ply")
        cameras_path = os.path.join(target_dir, "cams")
        scaled_cams_path = os.path.join(target_dir, "scaled_cams")
        output_dir = os.path.join(target_dir_path, "novel_views")
        
        # 필요한 디렉토리 생성 (존재하지 않을 경우)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(scaled_cams_path, exist_ok=True)
        
        # 메쉬 파일 로드
        mesh = o3d.io.read_triangle_mesh(input3_mesh_dir)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        colors_m = np.asarray(mesh.vertex_colors)
        normals_m = np.asarray(mesh.vertex_normals)

        # 이미지 이름 순서대로 정렬
        cameras_data, train_cams_data = load_camera_parameters(cameras_path, train_image_names, framework)

        # 필요한 경우 카메라 intrinsic 스케일링
        if dataset_type == "DTU":
            target_width = 400
            target_height = 300
        elif dataset_type == "LLFF":
            target_width = 504
            target_height = 378

        cameras_data, scale1 = scale_camera(cameras_data, target_width, target_height)
        train_cams_data, scale2 = scale_camera(train_cams_data, target_width, target_height)
        
        # Poisson Disk Sampling 후 저장
        print("Poison Disk Sampling...")
        sampled_points = mesh.sample_points_poisson_disk(number_of_points=n)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(sampled_points.points))
        pcd.colors = sampled_points.colors
        pcd.normals = sampled_points.normals
        # o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(os.path.join(output_dir, f"sampled_points_{n}.ply"), pcd)
        print(f"Sampled points saved to: {os.path.join(output_dir, f'sampled_points_{n}.ply')}")

        save_cams_convert_W2CtoC2W(cameras_data, scaled_cams_path)

        selected_cameras_data = [cam for cam in cameras_data
        if any(cam['img_name'].endswith(name) for name in train_image_names)]

        # new_camera setting 매개변수
        num_intermediate = 2
        vertical_amplitude = 0.15
        extrapolation_factor = 0.1
        num_vertical = 1
        
        new_cameras_data, intersection_point = generate_varied_camera_data_with_sphere(selected_cameras_data, num_intermediate=num_intermediate, vertical_amplitude=vertical_amplitude, extrapolation_factor=extrapolation_factor, num_vertical=num_vertical)
        
        # 각 카메라 정보에 대해 처리
        for camera_info in tqdm(new_cameras_data, desc=f"Processing Data: {target_data}", position=1, leave=False):
            w2c = np.array(camera_info['extrinsic'])
            k = camera_info['intrinsic']
            width = target_width
            height = target_height

            # W2C
            p_m, n_m, c_m, mask = world2Camera(w2c, k, vertices, normals_m, colors_m, camera_info['img_name'], framework)
            
            # 필터링된 정점 기반으로 새로운 mesh 생성
            filtered_vertices = p_m
            filtered_normals = n_m
            filtered_colors = c_m

            # 필터링된 메쉬 생성
            mesh_c = o3d.geometry.TriangleMesh()
            mesh_c.vertices = o3d.utility.Vector3dVector(filtered_vertices)
            mesh_c.vertex_normals = o3d.utility.Vector3dVector(filtered_normals)
            mesh_c.vertex_colors = o3d.utility.Vector3dVector(filtered_colors)
            mesh_c.triangles = o3d.utility.Vector3iVector(triangles)

            background = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
            projected_mesh, mask = project_and_draw_mesh(mesh_c, k, background)
            depth_img, depth_max, depth_min = mesh_depth(mesh, w2c, k, width, height)

            masked_img = np.zeros_like(projected_mesh)
            masked_img[mask != 1] = (255, 255, 255)

            image_name = camera_info['img_name']
            novel_view_path = os.path.join(output_dir, "images")
            novel_mask_path = os.path.join(output_dir, "masks")
            novel_depth_path = os.path.join(output_dir, "depths")

            os.makedirs(novel_view_path, exist_ok=True)
            os.makedirs(novel_mask_path, exist_ok=True)
            os.makedirs(novel_depth_path, exist_ok=True)

            cv2.imwrite(os.path.join(novel_view_path, f"{image_name}.png"), projected_mesh)
            cv2.imwrite(os.path.join(novel_mask_path, f"mask_{image_name}.png"),  masked_img)
            cv2.imwrite(os.path.join(novel_depth_path, f"depth_{image_name}.png"), depth_img)
            with open(os.path.join(novel_depth_path, f"depth_{image_name}.txt"), "w") as f:
                f.write(f"max: {depth_max}\n")
                f.write(f"min: {depth_min}\n")

        save_transposed_extrinsics(new_cameras_data, output_dir)

# ----------------------------------------------------------------------------------------------------------------------

# Frustum Culling
# World => Camera로 변환 후 Frustum에 해당하는 Mask return
def world2Camera(extrinsicParameter, intrinsicParameter, points, normals, colors, image_name, framework="dust3r"):
    fx = intrinsicParameter[0][0]
    fy = intrinsicParameter[1][1]
    width = intrinsicParameter[0][2] * 2
    height = intrinsicParameter[1][2] * 2
    znear = 0.01
    zfar = 50

    # World 좌표 => Camera 좌표
    # 1. Extrinsic Matrix에서 회전 및 평행 이동 추출
    # extrinsicParameter = np.linalg.inv(extrinsicParameter)
    R = extrinsicParameter[:3, :3]  # 회전 행렬
    t = extrinsicParameter[:3, 3]   # 평행 이동 벡터

    # 2. 회전 및 평행 이동을 사용해 카메라 좌표계로 변환 (R * (points - t))
    if framework == "dust3r":
        # Points는 (N, 3) -> Homogeneous 좌표계로 확장 (N, 4)
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)
        transformed_points_h = np.dot(extrinsicParameter, points_h.T).T  # (N, 4)

        # 변환된 Points의 x, y, z만 가져옴 (Homogeneous 좌표계 -> 3D로 변환)
        transformed_points = transformed_points_h[:, :3]

        # Normals 변환 (Extrinsic의 회전 부분만 사용)
        R = extrinsicParameter[:3, :3]  # 회전 행렬만 추출
        transformed_normals = np.dot(R, normals.T).T  # (N, 3)
    elif framework == "colmap":
        # 포인트에 회전과 평행 이동 적용: R * (points - t)
        transformed_points = np.dot(R, (points - t).T).T
        # 노멀에 회전만 적용: R * normals
        transformed_normals = np.dot(R, normals.T).T

    # View Frustum Culling
    # 카메라 좌표계에서 x, y, z 값 추출
    x, y, z = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]

    # # 카메라 좌표계 기준으로 point 저장
    # camera_pcd = o3d.geometry.PointCloud()
    # camera_pcd.points = o3d.utility.Vector3dVector(np.vstack((x, y, z)).T)
    # camera_pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.io.write_point_cloud(f"cam_coord_{image_name}.ply", camera_pcd)

    # FoV(시야각) 계산
    fovX = 2 * math.atan(width / (2 * fx))
    fovY = 2 * math.atan(height / (2 * fy))

    # # Frustum Culling 조건    
    # 1. 점들이 near와 far 평면 사이에 있는지 확인
    in_depth_range = (z >= znear) & (z <= zfar)

    # 2. 점들이 수평 시야각(FoV) 내에 있는지 확인
    x_bound = z * math.tan(fovX / 2)
    in_horizontal_fov = (x > -x_bound) & (x < x_bound)

    # 3. 점들이 수직 시야각(FoV) 내에 있는지 확인
    y_bound = z * math.tan(fovY / 2)
    in_vertical_fov = (y > -y_bound) & (y < y_bound)

    # 모든 조건 결합
    frustum_mask = in_depth_range & in_horizontal_fov & in_vertical_fov
    
    return transformed_points, transformed_normals, colors, frustum_mask


def project_and_draw_mesh(mesh, intrinsic, image):
    """
    메쉬를 2D 이미지에 투영하고 삼각형을 그립니다.
    
    Args:
        mesh: Open3D TriangleMesh 객체
        intrinsic: 카메라 내부 파라미터 (3x3 numpy 배열)
        image: OpenCV 이미지 배열 (H x W x 3)
    
    Returns:
        투영된 삼각형이 포함된 이미지
    """
    # 메쉬 정점과 삼각형 가져오기
    vertices = np.asarray(mesh.vertices)  # N x 3
    triangles = np.asarray(mesh.triangles)  # M x 3 (각 행이 정점 인덱스를 가짐)
    vertex_colors = np.asarray(mesh.vertex_colors)  # N x 3

    # 메쉬 정점 카메라 좌표에서 2D 이미지 좌표로 투영
    projected_points = np.dot(intrinsic, vertices.T).T
    projected_points[:, :2] /= projected_points[:, 2:3]  # 동차 좌표 정규화
    projected_points = projected_points[:, :2]  # u, v만 사용

    # 삼각형의 평균 깊이 계산
    triangle_depths = np.mean(vertices[triangles, 2], axis=1)  # 각 삼각형의 평균 z값 (M x 1)

    # 깊이를 기준으로 삼각형 정렬 (깊이가 큰 삼각형이 먼저 그려지도록 역순 정렬)
    sorted_indices = np.argsort(-triangle_depths)  # 깊이를 기준으로 내림차순 정렬
    triangles_sorted = triangles[sorted_indices]
    
    # 이미지 복사본 생성
    projected_image = image.copy()

    # 삼각형 그리기
    for triangle in tqdm(triangles_sorted, leave=False, desc="Drawing triangles"):
        # 삼각형 꼭짓점의 2D 좌표 가져오기
        pts_2d = projected_points[triangle].astype(int)  # 정수형 좌표 (3 x 2)

        # 이미지 범위 안에 있는지 확인
        if np.all(pts_2d[:, 0] >= 0) and np.all(pts_2d[:, 0] < image.shape[1]) and \
           np.all(pts_2d[:, 1] >= 0) and np.all(pts_2d[:, 1] < image.shape[0]):
            # 삼각형 색상 평균 계산 (정점 색상 평균)
            color = (vertex_colors[triangle].mean(axis=0) * 255).astype(int)
            color_rgb = tuple(map(int, color))
            color_bgr = color_rgb[::-1]  # OpenCV는 BGR 사용

            # 삼각형을 이미지에 그리기 (폴리곤)
            pts_2d = pts_2d.reshape((-1, 1, 2))  # OpenCV는 (N, 1, 2) 형태로 필요
            cv2.fillPoly(projected_image, [pts_2d], color=color_bgr)

    diff = cv2.absdiff(projected_image, image)
    mask = np.all(diff == 0, axis=-1).astype(np.uint8)

    return projected_image, mask


def mesh_depth(mesh, extrinsic, intrinsic, width, height):
    ext = np.array(extrinsic)
    R = ext[:3, :3]
    t = ext[:3, 3]
    
    R_C2W = R.T
    t_C2W = -R.T @ t
    
    eye = t_C2W
    center = eye + R_C2W[:, 2]
    up = -R_C2W[:, 1]
    
    print("Eye (Camera Position):", eye)
    print("Center (Look At):", center)
    print("Up (Up Direction):", up)
    print("Intrinsic Matrix:\n", intrinsic)
    
    renderer = None
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    renderer.scene.remove_geometry("mesh")  # 기존 메쉬 제거
    material = o3d.visualization.rendering.MaterialRecord()
    renderer.scene.add_geometry("mesh", mesh, material)
    
    renderer.scene.camera.set_projection(
        intrinsics=intrinsic,
        image_width=width,
        image_height=height,
        near_plane=0.1,
        far_plane=10.0
    )
    
    renderer.scene.camera.look_at(center=center, eye=eye, up=up)
    
    depth_image = renderer.render_to_depth_image()
    depth_array = np.asarray(depth_image)
    
    print("Raw Depth Array (Min, Max):", np.min(depth_array), np.max(depth_array))
    
    original_max = np.max(depth_array)
    original_min = np.min(depth_array)
    depth_map_normalized = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))
    scaled_depth_map = (depth_map_normalized * 255).astype(np.uint8)
    # depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    return scaled_depth_map, original_max, original_min


def generate_varied_camera_data_with_sphere(cameras_data, num_intermediate=3, vertical_amplitude=0.2, extrapolation_factor=0.15, num_vertical=1):
    """
    Duster 형식 카메라 정보를 기반으로 구 좌표계에서 중간 및 위아래 변형 카메라 생성.

    Args:
        cameras_data (list): 카메라 정보 리스트 (extrinsic, intrinsic 포함)
        num_intermediate (int): 두 카메라 사이에 생성할 중간 카메라 수
        vertical_amplitude (float): 상하 움직임 진폭 (구에서 위아래 움직임 범위)
        extrapolation_factor (float): 연장선 상의 가상 카메라 생성 비율 (0.0이면 보간만 수행)
        num_vertical (int): 각 보간 카메라에 대해 위아래 생성할 카메라 수

    Returns:
        list: 보간된 중간 및 위아래 변형된 카메라 정보 리스트
    """
    new_cameras = []
    
    intersection_point = estimate_intersection(cameras_data)

    for i in range(len(cameras_data) - 1):
        for t in np.linspace(-extrapolation_factor, 1 + extrapolation_factor, num_intermediate + 2):
            if 0 <= t <= 1 or extrapolation_factor > 0:  # 보간 및 외삽 포함
                base_camera = calculate_interpolated_camera_with_sphere(
                    cameras_data[i],
                    cameras_data[i + 1],
                    new_id=len(cameras_data) + len(new_cameras) + 500,
                    new_img_name=f"interpolated_{i}_{i+1}_t{t:.2f}",
                    t=t
                )
                base_camera = look_at(base_camera, intersection_point)
                new_cameras.append(base_camera)

                # 위아래 카메라 추가
                for v in np.linspace(-vertical_amplitude, vertical_amplitude, num_vertical):
                    vertical_camera = add_vertical_variation_to_camera(
                        base_camera,
                        v,
                        len(cameras_data) + len(new_cameras) + 500,
                        f"{base_camera['img_name']}_v{v:.2f}"
                    )
                    vertical_camera = look_at(vertical_camera, intersection_point)
                    new_cameras.append(vertical_camera)

    return new_cameras, intersection_point


def calculate_interpolated_camera_with_sphere(camera1, camera2, new_id, new_img_name, t):
    """
    Duster 형식 카메라 정보의 extrinsic 및 intrinsic을 구 좌표계를 기반으로 보간.

    Args:
        camera1 (dict): 첫 번째 카메라 정보
        camera2 (dict): 두 번째 카메라 정보
        new_id (int): 새롭게 생성될 카메라의 ID
        new_img_name (str): 새 이미지 이름
        t (float): 보간 비율 (t < 0 또는 t > 1이면 외삽 수행)

    Returns:
        dict: 보간된 새로운 카메라 정보
    """
    extr1 = np.array(camera1["extrinsic"])
    extr2 = np.array(camera2["extrinsic"])
    pos1 = extr1[:3, 3]
    pos2 = extr2[:3, 3]

    # 보간 위치 계산
    interpolated_position = ((1 - t) * pos1 + t * pos2)

    # 회전 보간
    rot1 = R.from_matrix(extr1[:3, :3])
    rot2 = R.from_matrix(extr2[:3, :3])
    if 0 <= t <= 1:
        # SLERP 사용
        slerp = Slerp([0, 1], R.from_matrix([rot1.as_matrix(), rot2.as_matrix()]))
        interpolated_rotation = slerp(t).as_matrix()
    else:
        # 외삽: 두 회전 행렬의 방향 벡터를 기반으로 직접 계산
        delta_rotation = rot2 * rot1.inv()
        extrapolated_rotation = delta_rotation ** t
        interpolated_rotation = (rot1 * extrapolated_rotation).as_matrix()

    new_extrinsic = np.eye(4)
    new_extrinsic[:3, :3] = interpolated_rotation
    new_extrinsic[:3, 3] = interpolated_position

    return {
        "id": new_id,
        "img_name": new_img_name,
        "extrinsic": new_extrinsic.tolist(),
        "intrinsic": camera1["intrinsic"]
    }


def add_vertical_variation_to_camera(camera, vertical_offset, new_id, new_img_name):
    """
    기존 카메라의 위치를 월드 좌표계 Y축 기준으로 이동 (회전은 유지).

    Args:
        camera (dict): 원본 카메라 정보
        vertical_offset (float): 수직 변형 값 (Y축 기준)
        new_id (int): 새 카메라 ID
        new_img_name (str): 새 이미지 이름

    Returns:
        dict: Y축 기준으로 위치가 이동된 새로운 카메라 정보
    """
    extr = np.array(camera["extrinsic"])
    pos = extr[:3, 3]  # 카메라 위치 추출 (Translation)

    # 새로운 위치 계산 (Y축 기준으로 이동)
    new_pos = pos.copy()
    new_pos[1] += vertical_offset  # Y축에 offset 추가

    # 새로운 Extrinsic 행렬 생성 (회전은 기존 값 유지)
    new_extrinsic = extr.copy()
    new_extrinsic[:3, 3] = new_pos

    return {
        "id": new_id,
        "img_name": new_img_name,
        "extrinsic": new_extrinsic.tolist(),
        "intrinsic": camera["intrinsic"]
    }


def load_camera_parameters(camera_dir, train_image_names, framework):
    """
    특정 이미지 목록에 맞는 카메라 정보를 읽어오고, 전체 및 특정 트레인 데이터를 반환합니다.

    Args:
        camera_dir (str): 카메라 정보가 저장된 폴더 경로.
        train_image_names (list): 사용할 이미지 이름 목록.
        framework (str): 사용 중인 프레임워크 (예: "dust3r").

    Returns:
        tuple: (전체 카메라 데이터 리스트, 트레인 카메라 데이터 리스트)
    """
    all_camera_data = []
    train_camera_data = []

    if framework == "dust3r":
        for filename in os.listdir(camera_dir):
            if filename.endswith("_cam.txt"):
                cam_file = os.path.join(camera_dir, filename)

                # 카메라 파일 읽기
                with open(cam_file, 'r') as f:
                    lines = f.readlines()

                # Extrinsic 매트릭스 추출
                try:
                    extrinsic_start = lines.index("extrinsic\n") + 1
                    extrinsic = np.array(
                        [[float(val) for val in line.split()] for line in lines[extrinsic_start:extrinsic_start + 4]]
                    )
                    extrinsic = np.linalg.inv(extrinsic)  # 역행렬 계산
                except (ValueError, IndexError):
                    print(f"Error parsing extrinsic for file: {filename}")
                    continue

                # Intrinsic 매트릭스 추출
                try:
                    intrinsic_start = lines.index("intrinsic\n") + 1
                    intrinsic = np.array(
                        [[float(val) for val in line.split()] for line in lines[intrinsic_start:intrinsic_start + 3]]
                    )
                except (ValueError, IndexError):
                    print(f"Error parsing intrinsic for file: {filename}")
                    continue

                img_name = filename.replace("_cam.txt", "")

                # 카메라 정보 저장
                camera_info = {
                    'img_name': img_name,
                    'extrinsic': extrinsic,
                    'intrinsic': intrinsic
                }

                all_camera_data.append(camera_info)

                # 트레인 데이터 필터링
                if img_name in train_image_names:
                    train_camera_data.append(camera_info)

    return all_camera_data, train_camera_data


def scale_camera(cameras_data, target_width, target_height):
    """
    카메라 데이터(cameras_data)의 intrinsic 정보를 target 크기에 맞게 스케일링합니다.

    Args:
        cameras_data (list): 카메라 정보 리스트 (intrinsic 포함).
        target_width (int): 새로운 이미지 가로 크기.
        target_height (int): 새로운 이미지 세로 크기.

    Returns:
        list: 스케일링된 cameras_data 리스트.
    """
    scaled_cameras = []
    scale_x = 1.0
    for camera in cameras_data:
        # 기존 intrinsic 행렬 가져오기
        extrinsic = np.array(camera['extrinsic'])
        intrinsic = np.array(camera['intrinsic'])

        # 기존 이미지 크기 추출 (cx, cy를 이용해 계산)
        original_width = intrinsic[0, 2] * 2  # cx의 2배
        original_height = intrinsic[1, 2] * 2  # cy의 2배

        # 스케일링 비율 계산
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        # intrinsic 행렬 스케일링
        intrinsic[0, 0] *= scale_x  # fx
        intrinsic[1, 1] *= scale_y  # fy
        intrinsic[0, 2] *= scale_x  # cx
        intrinsic[1, 2] *= scale_y  # cy
        
        # extrinsic[:3, 3] *= scale_x

        # 스케일링된 데이터를 새로운 카메라 데이터에 반영
        scaled_camera = camera.copy()
        scaled_camera['intrinsic'] = intrinsic.tolist()  # numpy 배열을 리스트로 변환
        scaled_camera['extrinsic'] = extrinsic.tolist()
        scaled_cameras.append(scaled_camera)

    return scaled_cameras, scale_x


def save_cams_convert_W2CtoC2W(cameras_data, output_dir):
    for cam in cameras_data:
        img_name = cam['img_name']
        extrinsic = cam['extrinsic']
        extrinsic = np.linalg.inv(extrinsic) # 변환해서 사용하던 행렬 다시 C2W로
        intrinsic = cam['intrinsic']
        
        output_path = os.path.join(output_dir, f"{img_name}_cam.txt")
        
        with open(output_path, 'w') as f:
            f.write("extrinsic\n")
            for row in extrinsic:
                f.write(" ".join([str(val) for val in row]) + "\n")
            
            f.write("\n")
            
            f.write("intrinsic\n")
            for row in intrinsic:
                f.write(" ".join([str(val) for val in row]) + "\n")


def save_transposed_extrinsics(new_cameras_data, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through cameras and transpose extrinsic
    for camera in new_cameras_data:
        extrinsic = np.array(camera["extrinsic"])  # 기존 extrinsic을 numpy 배열로 변환
        extrinsic = np.linalg.inv(extrinsic)  # 카메라 좌표계로 변환
        transposed_extrinsic = extrinsic.T  # Transpose 수행
        camera["extrinsic"] = transposed_extrinsic.tolist()  # Transposed extrinsic을 다시 리스트로 변환

    # Save the modified data as JSON
    with open(os.path.join(output_dir, "new_cameras.json"), 'w') as f:
        json.dump(new_cameras_data, f, indent=4)

# -----------------------------------------------------------------------
# Visualize 관련

def visualize_mesh_and_cameras(mesh, cameras_data, camera_scale=0.1):
    """
    Mesh와 카메라 좌표계를 Open3D로 시각화합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 시각화할 메쉬 객체
        cameras_data (list): 카메라 데이터 리스트 (new_camera_data)
        camera_scale (float): 카메라 좌표계의 스케일
    """
    vis_objects = []

    # Add mesh to the visualization
    mesh.compute_vertex_normals()
    vis_objects.append(mesh)

    # Visualize cameras
    for camera in cameras_data:
        # Extract camera position and rotation
        extrinsic = np.array(camera["extrinsic"])  # 4x4 행렬
        extrinsic = np.linalg.inv(extrinsic)  # 카메라 좌표계로 변환
        position = extrinsic[:3, 3]
        rotation = extrinsic[:3, :3]

        # Create a coordinate frame for the camera
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale, origin=position)
        
        # Apply the rotation to the coordinate frame
        camera_frame.rotate(rotation, center=camera_frame.get_center())
        vis_objects.append(camera_frame)

    # Visualize world coordinate frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis_objects.append(world_frame)

    # Start visualization
    o3d.visualization.draw_geometries(vis_objects, window_name="Mesh and Cameras")

def estimate_intersection(camera_poses):
    """
    Estimate the intersection point of rays from multiple cameras.
    
    Parameters:
    - camera_poses: List of 4x4 camera pose matrices (world to camera transformations)
    
    Returns:
    - intersection_point: 3D point where rays approximately intersect
    """
    A = []
    b = []
    
    for pose in camera_poses:
        # Extract camera center (translation) from pose
        pose = np.array(copy.deepcopy(pose["extrinsic"]))
        camera_center = pose[:3, 3]
        
        # Extract ray direction from rotation matrix (3rd column indicates z-axis direction)
        ray_direction = pose[:3, 2]  # Assuming the forward direction is along the z-axis
        
        # Normalize the ray direction to ensure it's a unit vector
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        # Construct the least squares system
        I = np.eye(3)
        A.append(I - np.outer(ray_direction, ray_direction))
        b.append((I - np.outer(ray_direction, ray_direction)) @ camera_center)
    
    # Stack A and b for solving Ax = b
    A = np.vstack(A)
    b = np.hstack(b)
    
    # Solve the least squares problem
    intersection_point = np.linalg.lstsq(A, b, rcond=None)[0]
    return intersection_point


def look_at(camera_pose, target_point):
    # 입력 extrinsic을 C2W로 변환
    extrinsic_w2c = np.array(camera_pose["extrinsic"])
    extrinsic_c2w = np.linalg.inv(extrinsic_w2c)
    
    camera_position = extrinsic_c2w[:3, 3]
    
    forward = target_point - camera_position
    forward /= np.linalg.norm(forward)
    up = np.array([0, 1, 0], dtype=np.float64)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, 0] = right
    rotation_matrix[:3, 1] = up
    rotation_matrix[:3, 2] = forward
    
    new_camera_pose_c2w = np.eye(4)
    new_camera_pose_c2w[:3, :3] = rotation_matrix[:3, :3]
    new_camera_pose_c2w[:3, 3] = camera_position
    
    # 결과를 다시 W2C 형태로 변환하여 저장
    new_camera_pose_w2c = np.linalg.inv(new_camera_pose_c2w)
    camera_pose["extrinsic"] = new_camera_pose_w2c.tolist()
    
    return camera_pose

# -----------------------------------------------------------------------

if __name__ == "__main__":
    main()
    