import os
import shutil
import random

DEBUG = True

# 📁 데이터셋 경로 설정
DATASET_DIR = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "1_1_800images")
LABELS_DIR = os.path.join(DATASET_DIR, "4_0_800size_txt_labels")

# 출력 폴더 (YOLO 요구 구조)
TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, "4_1_train_txt")
VAL_LABELS_DIR   = os.path.join(DATASET_DIR, "4_2_val_txt")
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "4_3_train_image")
VAL_IMAGES_DIR   = os.path.join(DATASET_DIR, "4_4_val_image")

# ⚖️ 데이터 분할 비율 (학습:검증 = 80:20)
TRAIN_RATIO = 0.8

# 📂 출력 디렉터리 생성
for dir_path in [TRAIN_LABELS_DIR, VAL_LABELS_DIR, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    if DEBUG:
        print(f"[DEBUG] 생성된 디렉터리: {dir_path}")

# 🔄 이미지와 라벨 매칭을 위한 확장자 설정
allowed_img_exts = (".jpg",)
# 이미지 파일들의 기본 이름(확장자 제외) 집합
image_files = set(os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR) if f.lower().endswith(allowed_img_exts))
# 라벨 파일들의 기본 이름(확장자 제외) 집합 (.txt)
label_files = set(os.path.splitext(f)[0] for f in os.listdir(LABELS_DIR) if f.lower().endswith(".txt"))
matched_files = list(image_files & label_files)

if DEBUG:
    print(f"[DEBUG] 총 이미지 파일: {len(image_files)}개, 총 라벨 파일: {len(label_files)}개")
    print(f"[DEBUG] 매칭된 파일 수: {len(matched_files)}개")

if not matched_files:
    raise ValueError("⚠️ 이미지와 라벨이 매칭된 파일이 없습니다. 파일 이름을 확인하세요.")

# 🔄 데이터 분할 (무작위 섞은 후 80%는 학습용, 20%는 검증용)
random.shuffle(matched_files)
train_count = int(len(matched_files) * TRAIN_RATIO)
train_files = matched_files[:train_count]
val_files = matched_files[train_count:]

if DEBUG:
    print(f"[DEBUG] 학습 데이터: {len(train_files)}개, 검증 데이터: {len(val_files)}개")

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

# 🚀 파일 복사 실행 (학습 데이터와 검증 데이터 각각)
copy_files(train_files, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
copy_files(val_files, VAL_IMAGES_DIR, VAL_LABELS_DIR)

print("✅ 데이터셋 분할 완료")
print(f" - 학습 데이터: {len(train_files)}개")
print(f" - 검증 데이터: {len(val_files)}개")
