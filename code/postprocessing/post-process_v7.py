import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from ultralytics import YOLO
from scipy.ndimage import label
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# 경로 설정
video_path = "/home/elicer/jny/video/input/train2.mp4"
output_path = "/home/elicer/jny/video/output/v7/train2_v7.mp4"
log_path = "/home/elicer/jny/video/output/v7/log.csv"
save_plot_dir = "/home/elicer/jny/video/output/v7/plots"
segformer_path = "/home/elicer/jhj/Segformer_Train/Final/result/checkpoint-3000"
yolo_model_path = "/home/elicer/syh/train_data/train(nc8)/weights/best.pt"

# 클래스 이름 및 범주 설정
class_names = [
    "Disabled Icon", "Women Icon", "No Parking Sign", "Traffic Cone",
    "Two-wheeled Vehicle", "Vehicle", "Human", 'Compact Car Icon'
]

object_categories = {
    "Reserved": [0, 7],
    "NoParking": [2, 3],
    "Vehicle": [4, 5],
    "Human": [6]
}


BACKGROUND_ID = 0
DRIVING_AREA_ID = 1
PARKING_AREA_ID = 2

parking_status_colors = {
    "Empty": (100, 255, 100),
    "Caution": (0, 140, 255),
    "Reserved": (255, 255, 0),
    "Blocked": (0, 0, 200)
}

segformer_colors = {
    0: (0, 0, 0),
    1: (200, 150, 255),
}

def get_category(class_id):
    for category, ids in object_categories.items():
        if class_id in ids:
            return category
    return "Other"

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO(yolo_model_path)
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
segformer_model = SegformerForSemanticSegmentation.from_pretrained(segformer_path).to(device)
segformer_model.eval()

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

log_data = []
performance_counter = defaultdict(Counter)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segformer_model(**inputs)
        logits = torch.nn.functional.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1)[0].cpu().numpy()

    color_mask = np.zeros_like(frame)
    color_mask[preds == DRIVING_AREA_ID] = segformer_colors[1]

    yolo_results = yolo_model(original)[0]
    boxes = yolo_results.boxes.data.cpu().numpy()

    empty_count = caution_count = reserved_count = blocked_count = 0
    risk_names = []

    parking_mask = (preds == PARKING_AREA_ID).astype(np.uint8)
    labeled_mask, num_labels = label(parking_mask)

    for i in range(1, num_labels + 1):
        region_mask = (labeled_mask == i).astype(np.uint8)
        if np.count_nonzero(region_mask) == 0:
            continue

        region_status = "Empty"

        for box in boxes:
            class_id = int(box[5])
            category = get_category(class_id)
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # 중심점 기반 조건
            in_region_by_center = (cy < h and cx < w and region_mask[cy, cx] == 1)

            # IoU 및 region 기준 조건 (NoParking, Human)
            iou_condition = False
            if category in ["NoParking", "Human"]:
                box_mask = np.zeros((h, w), dtype=np.uint8)
                box_mask[y1:y2, x1:x2] = 1
                intersection = np.logical_and(region_mask, box_mask).sum()
                region_area = region_mask.sum()
                box_area = (x2 - x1) * (y2 - y1)
                if box_area > 0 and (intersection / box_area > 0.02):
                    iou_condition = True
                elif region_area > 0 and (intersection / region_area > 0.02):
                    iou_condition = True

            # 중심점 or IoU 기반 조건 중 하나라도 만족하면 적용
            if not (in_region_by_center or iou_condition):
                continue

            # 일반 카테고리 분류 기준
            if category == "Reserved":
                region_status = "Reserved"
            elif category == "Human":
                region_status = "Caution"
            elif category in ["NoParking", "Vehicle"]:
                region_status = "Blocked"


        if region_status == "Empty":
            empty_count += 1
        elif region_status == "Reserved":
            reserved_count += 1
        elif region_status == "Blocked":
            blocked_count += 1
        elif region_status == "Caution":
            caution_count += 1

        for c in range(3):
            color_mask[:, :, c][region_mask == 1] = parking_status_colors[region_status][c]

        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_mask, contours, -1, (255, 255, 255), 2)

    overlay = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)
    edge_map = cv2.Canny(preds.astype(np.uint8), 50, 150)
    edges_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
    edges_rgb[np.where((edges_rgb != [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    final_overlay = cv2.addWeighted(overlay, 1.0, edges_rgb, 0.7, 0)

    for box in boxes:
        class_id = int(box[5])
        category = get_category(class_id)
        x1, y1, x2, y2 = map(int, box[:4])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(final_overlay, (cx, cy), 5, (255, 255, 255), -1)

        if category == "Reserved":
            cls_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            cv2.rectangle(final_overlay, (x1, y1), (x2, y2), (255, 255, 0), 3)
            text_size = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.rectangle(final_overlay, (x1, y2 + 10), (x1 + text_size[0], y2 + 10 + text_size[1]), (0, 0, 0), -1)
            cv2.putText(final_overlay, cls_name, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        if category in ["Vehicle", "Human"]:
            cls_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            cv2.rectangle(final_overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)
            text_size = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            cv2.rectangle(final_overlay, (x1, y2 + 15), (x1 + text_size[0], y2 + 15 + text_size[1]), (0, 0, 0), -1)
            cv2.putText(final_overlay, cls_name, (x1, y2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        if category == "NoParking":
            cls_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            cv2.rectangle(final_overlay, (x1, y1), (x2, y2), (0, 0, 200), 4)
            text_size = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            cv2.rectangle(final_overlay, (x1, y2 + 15), (x1 + text_size[0], y2 + 15 + text_size[1]), (0, 0, 0), -1)
            cv2.putText(final_overlay, cls_name, (x1, y2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2)

    risk_names = [class_names[int(box[5])] for box in boxes if get_category(int(box[5])) in ["Vehicle", "Human", "NoParking"]]
    risk_counts = Counter(risk_names)
    risk_str = ", ".join([f"{k}({v})" for k, v in risk_counts.items()]) or "None"

    log_data.append({
        "frame": frame_idx,
        "empty": empty_count,
        "caution": caution_count,
        "reserved": reserved_count,
        "blocked": blocked_count,
        "risks": risk_str
    })

    for box in boxes:
        class_id = int(box[5])
        category = get_category(class_id)
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        performance_counter[category][class_name] += 1

    cv2.rectangle(final_overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.putText(final_overlay, f"Frame: {frame_idx}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 4)
    cv2.putText(final_overlay, f"Risk: {risk_str}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 100, 255), 3)
    cv2.putText(
        final_overlay,
        f"Parking: Empty({empty_count}), Caution({caution_count}), Reserved({reserved_count}), Blocked({blocked_count})",
        (500, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 3
    )

    out.write(final_overlay)
    frame_idx += 1
    print(f"Processed frame {frame_idx}", end='\r')

cap.release()
out.release()

# CSV 로그 저장
pd.DataFrame(log_data).to_csv(log_path, index=False)
print(f"\n📄 CSV 로그 저장 완료: {log_path}")

# 성능 시각화
os.makedirs(save_plot_dir, exist_ok=True)
for category, counter in performance_counter.items():
    plt.figure(figsize=(10, 6))
    names = list(counter.keys())
    values = list(counter.values())
    plt.barh(names, values, color='skyblue')
    plt.title(f"{category} Category Frequency")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_plot_dir, f"{category}_performance.png"))
    plt.close()

print(f"📊 분류별 성능 시각화 완료: {save_plot_dir}")
print(f"✅ 영상 처리 완료: {output_path}")
