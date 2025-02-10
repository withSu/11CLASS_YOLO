import os
import shutil
import random
from PIL import Image

# 디버깅 모드 활성화
DEBUG = True

# 📁 데이터셋 경로 설정
DATASET_DIR = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset2"
IMAGES_DIR = os.path.join(DATASET_DIR, "1_2_800images")
LABELS_DIR = os.path.join(DATASET_DIR, "4_800labels")
OUTPUT_DIR = DATASET_DIR

# YOLO 요구 구조
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")

TRAIN_IMAGES = os.path.join(TRAIN_DIR, "images")
TRAIN_LABELS = os.path.join(TRAIN_DIR, "labels")
VAL_IMAGES = os.path.join(VAL_DIR, "images")
VAL_LABELS = os.path.join(VAL_DIR, "labels")

# ⚖️ 데이터 분할 비율
TRAIN_RATIO = 0.8

# 📂 디렉터리 생성
for dir_path in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
    os.makedirs(dir_path, exist_ok=True)
    if DEBUG:
        print(f"[DEBUG] 생성된 디렉터리: {dir_path}")

# 🔄 이미지와 라벨 매칭
allowed_img_exts = (".jpg",)
image_files = set(os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR) if f.lower().endswith(allowed_img_exts))
label_files = set(os.path.splitext(f)[0] for f in os.listdir(LABELS_DIR) if f.lower().endswith(".txt"))
matched_files = list(image_files & label_files)

if DEBUG:
    print(f"[DEBUG] 총 이미지 파일: {len(image_files)}, 총 라벨 파일: {len(label_files)}")
    print(f"[DEBUG] 매칭된 파일 수: {len(matched_files)}")

if not matched_files:
    raise ValueError("⚠️ 이미지와 라벨이 매칭된 파일이 없습니다. 파일 이름을 확인하세요.")

# 🔄 데이터 분할
random.shuffle(matched_files)
train_count = int(len(matched_files) * TRAIN_RATIO)
train_files = matched_files[:train_count]
val_files = matched_files[train_count:]

if DEBUG:
    print(f"[DEBUG] 학습 데이터: {len(train_files)}, 검증 데이터: {len(val_files)}")

# 📥 파일 복사 함수
def copy_files(files, image_dst, label_dst):
    for file in files:
        image_src = os.path.join(IMAGES_DIR, file + ".jpg")
        label_src = os.path.join(LABELS_DIR, file + ".txt")
        
        if os.path.exists(image_src) and os.path.exists(label_src):
            shutil.copy2(image_src, os.path.join(image_dst, file + ".jpg"))
            shutil.copy2(label_src, os.path.join(label_dst, file + ".txt"))
        else:
            print(f"⚠️ 누락된 파일: {file}")

# 🚀 파일 복사 실행
copy_files(train_files, TRAIN_IMAGES, TRAIN_LABELS)
copy_files(val_files, VAL_IMAGES, VAL_LABELS)

print("✅ 데이터셋 분할 완료")
print(f" - 학습 데이터: {len(train_files)}개")
print(f" - 검증 데이터: {len(val_files)}개")
