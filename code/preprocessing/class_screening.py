import os

# 클래스 매핑
class_mapping = {
    5: 4,  
    6: 5,  
    7: 6,  
    8: 7,   
    9: 8, 
    10:9,
    11:10,
    12:11,
    13:12
}



# 경로 설정
old_label_dir = '/home/elicer/dataset/jny_dataset/labels_yolo/old_val'
new_label_dir = '/home/elicer/dataset/jny_dataset/labels_yolo/val'
os.makedirs(new_label_dir, exist_ok=True)

# 변환
for fname in os.listdir(old_label_dir):
    if not fname.endswith('.txt'):
        continue

    old_path = os.path.join(old_label_dir, fname)
    new_path = os.path.join(new_label_dir, fname)

    new_lines = []
    with open(old_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue  
            try:
                class_id = int(parts[0])
            except ValueError:
                print(f"❗ 잘못된 라벨 '{parts[0]}' → 건너뜀: {fname}")
                continue

            if class_id in class_mapping:
                new_class_id = class_mapping[class_id]
                new_line = ' '.join([str(new_class_id)] + parts[1:])
                new_lines.append(new_line)

    # 새로운 라벨만 저장
    if new_lines:
        with open(new_path, 'w') as f:
            f.write('\n'.join(new_lines))
