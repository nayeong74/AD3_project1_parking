import os
import cv2
import numpy as np

# 경로 설정
label_dir = "/home/elicer/dataset/labels_yolo/train"
image_dir = "/home/elicer/dataset/images/train"
save_dir = "/home/elicer/dataset/visualized_images"
os.makedirs(save_dir, exist_ok=True)

target_class_id = 1  # 확인할 클래스 ID
max_images = 10      # 최대 저장할 이미지 수
saved_count = 0      # 저장된 이미지 개수

# 시각화 시작
for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(label_dir, label_file)

    with open(label_path, "r") as f:
        lines = f.readlines()

    polygons = []
    for line in lines:
        parts = line.strip().split()
        if int(parts[0]) != target_class_id:
            continue

        coords = list(map(float, parts[1:]))
        points = [(int(coords[i] * 1280), int(coords[i + 1] * 720)) for i in range(0, len(coords), 2)]
        polygons.append(points)

    if not polygons:
        continue  # class id 1이 없는 경우 skip

    # 이미지 불러오기
    image_name = label_file.replace(".txt", ".png")
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"이미지 {image_path} 로드 실패")
        continue

    # polygon 시각화
    for poly in polygons:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=3)

    # 이미지 저장
    save_path = os.path.join(save_dir, image_name)
    cv2.imwrite(save_path, image)
    saved_count += 1
    print(f"저장 완료: {save_path}")

    if saved_count >= max_images:
        print(f"\n✅ {max_images}개 이미지 저장 완료. 종료합니다.")
        break
