import os

# 경로 설정
label_dir = "/home/elicer/jny/dataset/labels"
image_dir = "/home/elicer/jny/dataset/images"

# 라벨 파일 정렬 (시간순)
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])

deleted_count = 0

for i, label_file in enumerate(label_files):
    if i % 4 != 0:
        # 라벨 삭제
        label_path = os.path.join(label_dir, label_file)
        if os.path.exists(label_path):
            os.remove(label_path)
            deleted_count += 1

        # 대응하는 이미지 삭제 
        base_name = os.path.splitext(label_file)[0]
        image_path = os.path.join(image_dir, base_name + ".png")
        if os.path.exists(image_path):
            os.remove(image_path)
            deleted_count += 1

print(f"총 {deleted_count}개 파일 삭제 완료 (라벨 + 이미지 포함).")
