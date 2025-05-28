import os

# 기준 경로
base_path = r"/home/elicer/181.실내_자율주차용_데이터/01-1.정식개방데이터/Training"

# 원천데이터 및 라벨링데이터 경로
camera_root = os.path.join(base_path, "01.원천데이터")
label_root = os.path.join(base_path, "02.라벨링데이터")

# 하위 폴더 이름 리스트 (날짜/시간 매칭용 키 추출)
def extract_key(name):
    parts = name.split("_")
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit(): 
            return "_".join(parts[i:i+2])  
    return None

# 폴더명에서 key 추출
camera_folders = os.listdir(camera_root)
label_folders = os.listdir(label_root)

camera_keys = set(filter(None, [extract_key(f) for f in camera_folders]))
label_keys = set(filter(None, [extract_key(f) for f in label_folders]))

# 비교
only_in_camera = sorted(camera_keys - label_keys)
only_in_label = sorted(label_keys - camera_keys)
matched = sorted(camera_keys & label_keys)

# 결과 출력
print("✅ 날짜/시간 키 일치 여부")
print(f"- 매칭된 세트: {len(matched)}개")
print(f"- 원천데이터에만 있는 날짜: {only_in_camera}")
print(f"- 라벨링데이터에만 있는 날짜: {only_in_label}")
