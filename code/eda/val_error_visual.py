from ultralytics import YOLO
import numpy as np
import os

# 모델 로드
model = YOLO('/home/elicer/jny/yolov8m_seg_v1/weights/best.pt')

# 예측 실행 
results = model.predict(
    source='/home/elicer/try1/dataset/images/val',  # 이미지 경로
    imgsz=1024,
    save=True,
    save_txt=False,
    conf=0.25,
    iou=0.5,
    name='yolov8m_seg_v1_val',
    project='/home/elicer/jny',
    stream=True  
)

# 저장 디렉토리 생성
save_dir = '/home/elicer/jny/yolov8m_seg_v1_val'
os.makedirs(save_dir, exist_ok=True)

# 결과에서 마스크 저장
for i, pred in enumerate(results):
    if hasattr(pred, 'masks') and pred.masks is not None:
        masks = pred.masks.data.cpu().numpy()  # (n, h, w)
        np.save(os.path.join(save_dir, f"masks_{i}.npy"), masks)

