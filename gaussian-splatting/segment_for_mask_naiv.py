import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1) 환경 설정
# -------------------------------------------------------
target = "scan110"  # 예: "scan8", "scan110" 등
data_path = f'/home/lsw/MPGS/data/DTU/{target}'
sparse_num = 3
image_path = os.path.join(data_path, 'dust3r', 'images')
mask_path = os.path.join(data_path, 'masks')
os.makedirs(mask_path, exist_ok=True)

# sparse_n.txt 에는 특정 이미지 인덱스들이 적혀 있다고 가정
sparse_ids = np.loadtxt(os.path.join(data_path, f'sparse_{sparse_num}.txt'), dtype=np.int32)

# 해당 인덱스들에 해당하는 이미지 파일명만 추출
all_image_names = sorted(os.listdir(image_path))
image_names = [name for idx, name in enumerate(all_image_names) if idx in sparse_ids]

# -------------------------------------------------------
# 2) 이미지 로드 (OpenCV + RGB 변환)
# -------------------------------------------------------
images = [
    cv2.cvtColor(cv2.imread(os.path.join(image_path, image_name)), cv2.COLOR_BGR2RGB)
    for image_name in image_names
]

# -------------------------------------------------------
# 3) 배경(검정 + 흰 탁자) 제거 후 물체 마스크 생성
#    - "검정 배경" 판단: 모든 채널 < black_thresh
#    - "하얀 탁자" 판단: 모든 채널 > white_thresh
#    - 두 마스크 OR => 전체 배경 마스크
#    - invert => 물체만 마스크
#    - morphology(열림/닫힘)으로 노이즈 제거 & 구멍 메우기
# -------------------------------------------------------

# scan110 등 특이 케이스에 따라 threshold 조정 가능
if "scan110" in target:
    black_thresh  = 15
    white_thresh  = 220
else:
    black_thresh  = 30
    white_thresh  = 200

for image_name, image in zip(image_names, images):
    # image: shape (H, W, 3), 값 범위 [0..255]

    # 1) 검은 배경 마스크
    #    모든 채널이 black_thresh 보다 작으면 True
    mask_black = np.all(image < black_thresh, axis=2)

    # 2) 흰 탁자 마스크
    #    모든 채널이 white_thresh 보다 크면 True
    mask_white = np.all(image > white_thresh, axis=2)

    # 3) 전체 배경 마스크 (검정 or 하양)
    #    mask_black, mask_white 둘 중 하나라도 True면 배경
    mask_bg = mask_black | mask_white

    # 4) 물체 마스크(전경) = 배경 마스크의 반전
    #    True(=배경)은 False, False(=물체)는 True
    mask_obj = np.logical_not(mask_bg)

    # 5) [옵션] 미세한 노이즈 제거를 위해 morphological operation 수행
    #    - 열림(cv2.MORPH_OPEN): 작은 잡음 제거
    #    - 닫힘(cv2.MORPH_CLOSE): 구멍 메우기
    mask_255 = (mask_obj * 255).astype(np.uint8)

    # 커널 크기는 상황에 맞춰 조절(작으면 잡음제거에 부족, 너무 크면 디테일이 사라짐)
    kernel_open = np.ones((2, 2), np.uint8)
    kernel_close = np.ones((15, 15), np.uint8)

    # 먼저 열림으로 (배경 속 작은 흰 점) 제거
    mask_opened = cv2.morphologyEx(mask_255, cv2.MORPH_OPEN, kernel_open)

    # 닫힘으로 (물체 내부의 검은 구멍) 제거
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)
    # ---------------------------------------------------
    # 6) 추가: Contour로 구멍 채우기
    # ---------------------------------------------------
    #    - 모든 contour를 찾고,
    #    - 구멍(자식) contour만 선택해서 메우기 (혹은 전체 contour 메우기)
    contours, hierarchy = cv2.findContours(mask_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # hierarchy 구조: [ [next, prev, child, parent], ... ]
    # 자식(child)이 있으면 구멍인 경우가 많다.

    # 방법 A) 간단히 모든 contour를 한 번에 채우기
    # for cnt in contours:
    #     cv2.drawContours(mask_closed, [cnt], 0, 255, -1)

    # 방법 B) hierarchy를 확인하여 "부모가 있는 contour"만 채우기
    if hierarchy is not None:
        for idx, cnt in enumerate(contours):
            # hierarchy[0][idx][3] != -1 이면 '부모'가 존재한다 => 내부 구멍인 경우가 많음
            if hierarchy[0][idx][3] != -1:
                cv2.drawContours(mask_closed, [cnt], 0, 255, -1)

    # 6) 최종 마스크를 0/255 형태로 저장

    save_path = os.path.join(mask_path, image_name)
    cv2.imwrite(save_path, mask_closed)

    print(f"[INFO] Saved mask => {save_path}")

print("[DONE] All masks saved.")
