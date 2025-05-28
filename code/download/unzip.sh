#!/bin/bash

echo "📦 ZIP 파일 압축 해제 시작..."

# 현재 디렉토리에서 .zip 파일들을 찾아 반복
for zipfile in *.zip; do
  folder="${zipfile%.zip}" # zip 확장자 제거하여 폴더 이름 생성
  echo "🔍 압축 해제 중: $zipfile → $folder/"
  mkdir -p "$folder"
  unzip -o "$zipfile" -d "$folder" > /dev/null
  echo "✅ 완료: $zipfile"
  rm "$zipfile"
done

echo "🎉 모든 zip 파일 압축 해제 완료!"