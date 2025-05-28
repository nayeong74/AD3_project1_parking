import os
import matplotlib.pyplot as plt
from collections import Counter

# 경로 설정
label_dir = "/home/elicer/jny/dataset/labels/train"  

# 클래스 이름 정의 
class_names = [ 
    'Parking Area Number',   # 0
    'Sign',                  # 1
    'Disabled Icon',         # 2
    'Women Icon',            # 3
    'No Parking Sign',       # 4
    'Traffic Cone',          # 5
    'Fire Extinguisher',     # 6
    'Undefined Object',      # 7
    'Two-wheeled Vehicle',   # 8
    'Vehicle',               # 9
    'Wheelchair',            # 10
    'Stroller',              # 11
    'Shopping Cart',         # 12
    'Human'                  # 13          
]

# 클래스별 개수 집계
counter = Counter()
for fname in os.listdir(label_dir):
    if fname.endswith(".txt"):
        with open(os.path.join(label_dir, fname), 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue
                class_id = int(line.split()[0])
                counter[class_id] += 1

# 클래스별 개수 정렬
sorted_counts = sorted(counter.items())
class_ids, counts = zip(*sorted_counts)
class_labels = [class_names[i] for i in class_ids]

# 시각화
plt.figure(figsize=(10, 5))
plt.bar(class_labels, counts, color='skyblue')
plt.xticks(rotation=45)
plt.title("Class Distribution in Training Dataset")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()
