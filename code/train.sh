# try1 baseline
yolo task=segment \
  mode=train \
  model=yolov8n-seg.pt \
  data=/home/elicer/try1/dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  name=yolov8n_seg_filtered_v2 \
  project=/home/elicer/jny \
  plots=True
  
yolo task=segment \
  mode=val \
  model=/home/elicer/jny/yolov8n_seg_filtered_v2/weights/best.pt \
  data=/home/elicer/try1/dataset/data.yaml \
  imgsz=640 \
  batch=16 \
  name=yolov8n_seg_filtered_v2_val \
  project=/home/elicer/jny \
  plots=True

#try1-ver2
yolo task=segment \
  mode=train \
  model=yolov8m-seg.pt \
  data=/home/elicer/jny/dataset/data.yaml \
  epochs=100 \
  imgsz=1024 \
  batch=1 \
  name=yolov8m_seg_v2\
  project=/home/elicer/jny \
  plots=True \
  save=True

#yolov8m_seg_object
yolo task=segment \
  mode=train \
  model=yolov8m-seg.pt \
  data=/home/elicer/jny/dataset/data.yaml \
  epochs=100 \
  imgsz=1024 \
  batch=4 \
  name=yolov8m_seg_object \
  project=/home/elicer/jny/result \
  plots=True \
  save=True \
  cache=True

#yolov8m_seg_object_v2
yolo task=segment \
  mode=train \
  model=yolov8m-seg.pt \
  data=/home/elicer/jny/dataset/data.yaml \
  epochs=150 \
  imgsz=1024 \
  batch=4 \
  name=yolov8m_seg_object_v2 \
  project=/home/elicer/jny/result \
  plots=True \
  save=True \
  cache=True

#yolov8m_seg_object_v3
yolo task=detect \
  mode=train \
  model=yolov8x.pt \
  data=/home/elicer/dataset/data_yolo.yaml \
  epochs=100 \
  imgsz=1024 \
  batch=4 \
  lr0=0.001 \
  warmup_epochs=3 \
  close_mosaic=10 \
  scale=0.4 translate=0.1 \
  fliplr=0.5 \
  mosaic=1.0 \
  copy_paste=0.1 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4\
  name=yolov8x_box_object_v1 \
  project=/home/elicer/jny/result \
  plots=True \
  save=True \
  cache=True
