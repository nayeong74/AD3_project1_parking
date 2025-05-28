import os
import numpy as np

def convert_polygon_to_box_txt(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        if not fname.endswith(".txt"):
            continue

        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        new_lines = []

        with open(src_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                # box 형식이면 그대로 사용
                if len(parts) == 5:
                    new_lines.append(line.strip())
                    continue

                class_id = parts[0]
                polygon = list(map(float, parts[5:]))  # polygon 좌표만 추출

                # polygon을 x, y 좌표로 나누기
                xs = polygon[0::2]
                ys = polygon[1::2]

                # bounding box 계산
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                new_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                new_lines.append(new_line)

        with open(dst_path, "w") as f:
            f.write("\n".join(new_lines))

    print(f"✅ 변환 완료: {dst_dir} 에 저장되었습니다.")

# 사용 예시
convert_polygon_to_box_txt(
    src_dir="/home/elicer/dataset/labels/val",
    dst_dir="/home/elicer/dataset/labels_box/val"
)
