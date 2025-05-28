import os
import json
from pathlib import Path

# 클래스 정의
class_names = [
    'Undefined Stuff', 'Wall', 'Driving Area', 'Non Driving Area', 'Parking Line',
    'Parking Area', 'No Parking Area', 'Big Notice', 'Pillar', 'Parking Area Number',
    'Disabled Icon', 'Women Icon', 'Compact Car Icon', 'Speed Bump', 'Parking Block',
    'Billboard', 'Toll Bar', 'Sign', 'No Parking Sign', 'Traffic Cone',
    'Fire Extinguisher', 'Undefined Object', 'Two-wheeled Vehicle', 'Vehicle',
    'Wheelchair', 'Stroller', 'Shopping Cart', 'Animal', 'Human'
]
class_to_id = {name: i for i, name in enumerate(class_names)}

# 경로 설정
root_labels = Path(r"/home/elicer/jny/dataset/oldd_labels")
output_labels = Path(r"/home/elicer/jny/dataset/old_labels")
output_labels.mkdir(parents=True, exist_ok=True)

# 변환 함수
def convert_segmentation_json_to_yoloseg(json_path, output_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        width = data.get("meta", {}).get("size", {}).get("width", 1)
        height = data.get("meta", {}).get("size", {}).get("height", 1)

        lines = []
        for obj in data.get("objects", []):
            class_id = class_to_id.get(obj["class_name"])
            if class_id is None:
                print(f"⚠️ 알 수 없는 클래스: {obj['class_name']}")
                continue

            for polygon in obj.get("annotation", []):
                coords = []
                for pt_list in polygon:
                    for pt in pt_list:
                        if isinstance(pt, dict) and "x" in pt and "y" in pt:
                            x = pt["x"] / width
                            y = pt["y"] / height
                            coords.extend([f"{x:.6f}", f"{y:.6f}"])
                if coords:
                    line = f"{class_id} " + " ".join(coords)
                    lines.append(line)

        if lines:
            with open(output_path, 'w') as f:
                f.write("\n".join(lines))

    except Exception as e:
        print(f"❌ 오류 발생: {json_path} - {e}")

# 하위 폴더에 있는 json 처리
for folder in root_labels.glob("*"):
    if folder.is_dir():
        output_subdir = output_labels / folder.name
        output_subdir.mkdir(parents=True, exist_ok=True)

        for json_file in folder.glob("*.json"):
            output_txt = output_subdir / (json_file.stem + ".txt")
            convert_segmentation_json_to_yoloseg(json_file, output_txt)

# 루트 labels 폴더에 직접 있는 json도 처리
for json_file in root_labels.glob("*.json"):
    output_txt = output_labels / (json_file.stem + ".txt")
    convert_segmentation_json_to_yoloseg(json_file, output_txt)

print("✅ YOLO-SEG 학습용 포맷으로 변환 완료.")
