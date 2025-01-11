import os
import cv2
import numpy as np

def combine_and_save_images(dataset_dir, output_dir):
    """
    주어진 데이터셋 디렉토리에서 train과 test 폴더 내의 모든 반복(iter)에 대해 이미지를 합쳐 저장합니다.

    Args:
        dataset_dir (str): 데이터셋의 상위 디렉토리 경로.
        output_dir (str): 합쳐진 이미지를 저장할 디렉토리 경로.
    """
    # train과 test 경로 설정
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    # output 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    for mode, mode_dir in zip(["train", "test"], [train_dir, test_dir]):
        # 모든 ours_{iter} 디렉토리를 탐색
        iter_dirs = [os.path.join(mode_dir, d) for d in os.listdir(mode_dir) if d.startswith("ours_")]

        for iter_dir in iter_dirs:
            if not os.path.exists(iter_dir):
                print(f"Warning: {iter_dir} does not exist.")
                continue

            # 반복 디렉토리 출력 경로 설정
            iter_output_dir = os.path.join(output_dir, mode, os.path.basename(iter_dir))
            os.makedirs(iter_output_dir, exist_ok=True)

            # 자식 디렉토리 이름
            sub_dirs = ["gt", "mask", "masked_rendering", "renders"]

            # 각 자식 디렉토리 경로
            sub_dir_paths = [os.path.join(iter_dir, sub_dir) for sub_dir in sub_dirs]

            # 다섯자리 숫자의 이름을 가지는 이미지 파일 이름 수집
            if not os.path.exists(sub_dir_paths[0]):
                print(f"Warning: {iter_dir} does not contain required subdirectories.")
                continue

            file_names = sorted(os.listdir(sub_dir_paths[0]))

            for file_name in file_names:
                if not file_name.endswith(('.png', '.jpg', '.jpeg')):
                    continue

                # 각 디렉토리에서 이미지를 읽기
                images = []
                for sub_dir, label in zip(sub_dir_paths, sub_dirs):
                    image_path = os.path.join(sub_dir, file_name)
                    if os.path.exists(image_path):
                        img = cv2.imread(image_path)
                        if img is not None:
                            # 이미지 상단에 레이블 추가
                            height, width = img.shape[:2]
                            label_img = np.zeros((30, width, 3), dtype=np.uint8)
                            label_img[:] = (255, 255, 255)
                            cv2.putText(label_img, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                            img = np.vstack((label_img, img))
                            images.append(img)
                        else:
                            print(f"Warning: Unable to read image {image_path}")
                    else:
                        print(f"Warning: File not found {image_path}")

                # 모든 이미지를 가로로 합치기
                if images:
                    combined_image = np.hstack(images)

                    # 합쳐진 이미지를 저장
                    output_path = os.path.join(iter_output_dir, file_name)
                    cv2.imwrite(output_path, combined_image)
                    print(f"Saved combined image to {output_path}")
                else:
                    print(f"Warning: No valid images to combine for {file_name}")

# Example usage
data = "scan21_novel48"
dataset_dir = f"/home/airlabs/SuGaR/gaussian_splatting/output/DTU/{data}"
output_dir = f"/home/airlabs/SuGaR/gaussian_splatting/output/DTU/{data}/overview"
combine_and_save_images(dataset_dir, output_dir)
