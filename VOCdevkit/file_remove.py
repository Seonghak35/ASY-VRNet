import os

# 디렉토리 경로 설정
image_dir = './VOC2007/JPEGImages'  # 예: './images'
remove_dir = './VOC2007/radar'  # 예: './removes'

# 이미지 이름 리스트 (확장자 제거)
image_names = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')}

for remove_file in os.listdir(remove_dir):
    if remove_file.endswith('.csv'):
        remove_name = os.path.splitext(remove_file)[0]
        if remove_name not in image_names:
            remove_path = os.path.join(remove_dir, remove_file)
            os.remove(remove_path)
            print(f"Deleted: {remove_path}")
