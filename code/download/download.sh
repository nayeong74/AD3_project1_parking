#!/bin/bash

# AIHub API 인증키 입력
API_KEY="D454F164-BF4F-4F42-9BB7-629EE2CB5DB1"
DATASET_KEY=71576

# filekey 목록 
FILE_KEYS=(
    475771
)

# 다운로드 실행
for FILE_KEY in "${FILE_KEYS[@]}"; do
  echo "📥 Downloading camera filekey: $FILE_KEY ..."
  aihubshell -mode d -datasetkey $DATASET_KEY -filekey $FILE_KEY -aihubapikey $API_KEY
done

echo "✅ 데이터 다운로드 완료!"