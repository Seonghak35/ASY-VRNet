import os
import shutil

# 설정
source_folder = './detection/xml'       # 원본 폴더 경로
destination_folder = './waterscenes_sample_1000/detection/xml'  # 복사 대상 폴더 경로
file_extension = '.xml'  # 복사할 파일의 확장자 (예: .png, .txt, .json 등)

# 목적지 폴더가 없으면 생성
os.makedirs(destination_folder, exist_ok=True)

# 00001 ~ 01000까지 반복
for i in range(1, 1001):
    filename = f"{i:05d}{file_extension}"  # '00001.jpg' 형식
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(destination_folder, filename)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"파일 없음: {src_path}")