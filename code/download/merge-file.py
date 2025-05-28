import os
import shutil
from pathlib import Path

# 경로 설정
labels_root = Path("/home/elicer/jny/dataset/images")

# 하위 폴더 순회
for subfolder in labels_root.iterdir():
    if subfolder.is_dir():
        for json_file in subfolder.glob("*.png"):
            dest_file = labels_root / json_file.name
            if not dest_file.exists():
                shutil.move(str(json_file), str(dest_file))
            else:
                print(f"⚠️ 중복 파일로 이동 생략: {json_file.name}")
        
        # 폴더가 비었으면 삭제
        try:
            subfolder.rmdir()
            print(f"🗑️ 빈 폴더 삭제: {subfolder.name}")
        except OSError:
            print(f"📁 폴더에 다른 파일이 남아 있어 삭제 생략: {subfolder.name}")

print("✅ 모든 파일 상위 폴더로 이동 완료.")
