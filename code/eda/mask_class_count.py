from collections import Counter
import os
import numpy as np
from PIL import Image

mask_dir = "/home/elicer/jny/dataset/masks/train"
counter = Counter()

for fname in os.listdir(mask_dir):
    if fname.endswith(".png"):
        mask = np.array(Image.open(os.path.join(mask_dir, fname)), dtype=np.uint8)
        counter.update(mask.flatten().tolist())

print("클래스별 픽셀 수:", counter)
