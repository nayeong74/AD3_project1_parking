import os

# 기준 경로
base_path = r"/home/elicer/181.실내_자율주차용_데이터/01-1.정식개방데이터/Training"
camera_root = os.path.join(base_path, "01.원천데이터")
label_root = os.path.join(base_path, "02.라벨링데이터")

# 파일 이름에서 공통 키 추출 (확장자 제거)
def extract_file_key(filename):
    return os.path.splitext(filename)[0]

# 날짜/시간 키 추출
def extract_folder_key(name):
    parts = name.split("_")
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit():  # ex: 20220929
            return "_".join(parts[i:i+2])
    return None

# 전체 비교 로직
camera_dirs = os.listdir(camera_root)
segment_dirs = [f for f in os.listdir(label_root) if f.endswith(".segmentation")]

matched_keys = set([extract_folder_key(f) for f in camera_dirs]) & \
               set([extract_folder_key(f) for f in segment_dirs]) 

print("📂 총 비교 대상 폴더:", len(matched_keys))
print("------------------------------------------------------------")

for key in sorted(matched_keys):
    cam_dir = next((d for d in camera_dirs if key in d), None)
    seg_dir = next((d for d in segment_dirs if key in d), None)

    cam_path = os.path.join(camera_root, cam_dir)
    seg_path = os.path.join(label_root, seg_dir)

    cam_files = set(extract_file_key(f) for f in os.listdir(cam_path) if f.endswith(".png"))
    seg_files = set(extract_file_key(f) for f in os.listdir(seg_path) if f.endswith(".json"))

    # 차이 계산
    seg_missing = cam_files - seg_files

    print(f"📁 {key}")
    print(f"  - 카메라 프레임 수: {len(cam_files)}")
    print(f"  - 세그먼트 누락: {len(seg_missing)}개")
    if seg_missing:
        print(f"    ⮩ 세그먼트 누락 예시: {sorted(list(seg_missing))[:3]}")
    print("-" * 60)
