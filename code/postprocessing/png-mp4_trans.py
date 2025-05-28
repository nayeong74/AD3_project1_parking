import cv2
import os
from glob import glob

# 입력 이미지 경로 설정
image_folder = '/home/elicer/jny/video/images'
output_video_path = '/home/elicer/jny/video/input_video.mp4'
fps = 30  # 초당 프레임 수

# PNG 파일들을 정렬된 순서로 불러오기
images = sorted(glob(os.path.join(image_folder, '*.png')))

# 첫 이미지의 크기 확인
sample_image = cv2.imread(images[0])
height, width, _ = sample_image.shape

# 비디오 인코더 설정 및 VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 'XVID', 'avc1'
video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 이미지들을 비디오로 쓰기
for img_path in images:
    frame = cv2.imread(img_path)
    video.write(frame)

# 리소스 해제
video.release()
print(f'비디오 저장 완료: {output_video_path}')
