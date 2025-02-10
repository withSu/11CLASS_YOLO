import os
from PIL import Image

# ======= 데이터셋 경로 설정 =======
DATASET_DIR = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

TRAIN_IMAGES = os.path.join(TRAIN_DIR, "images")
TRAIN_LABELS = os.path.join(TRAIN_DIR, "labels")
VAL_IMAGES = os.path.join(VAL_DIR, "images")
VAL_LABELS = os.path.join(VAL_DIR, "labels")

def check_integrity(image_dir, label_dir):
    """
    1. 이미지와 라벨 디렉터리 내의 파일 이름(확장자 제거)이 올바르게 대응하는지 확인합니다.
    2. 각 이미지 파일을 Pillow의 load()를 통해 메모리로 완전히 로드할 수 있는지(손상 여부)를 체크합니다.
    """
    allowed_img_exts = ('.jpg', '.jpeg', '.png')
    
    # 이미지와 라벨 파일 목록 수집
    image_list = [f for f in os.listdir(image_dir) if f.lower().endswith(allowed_img_exts)]
    label_list = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]
    
    # 기본 이름(확장자 제거) 추출
    image_basenames = {os.path.splitext(f)[0] for f in image_list}
    label_basenames = {os.path.splitext(f)[0] for f in label_list}
    
    print(f">> 검증 중: {image_dir} 와 {label_dir}")
    print(f" - 이미지 파일 수: {len(image_list)} (기본 이름: {len(image_basenames)}개)")
    print(f" - 라벨 파일 수: {len(label_list)} (기본 이름: {len(label_basenames)}개)")
    
    # 이름 매칭 체크
    missing_labels = image_basenames - label_basenames
    missing_images = label_basenames - image_basenames
    
    if missing_labels:
        print("‼️ 아래 이미지에 대응하는 라벨 파일이 없습니다:")
        for name in sorted(missing_labels):
            print("  -", name)
    if missing_images:
        print("‼️ 아래 라벨에 대응하는 이미지 파일이 없습니다:")
        for name in sorted(missing_images):
            print("  -", name)
    if not missing_labels and not missing_images:
        print("✅ 이미지와 라벨의 파일 이름 매칭이 정확합니다.")
    
    # 이미지 무결성 체크
    integrity_errors = []
    for image_file in image_list:
        image_path = os.path.join(image_dir, image_file)
        try:
            with Image.open(image_path) as img:
                img.load()
        except Exception as e:
            integrity_errors.append(f"이미지 로드 실패: {image_path} (에러: {e})")
    
    if integrity_errors:
        print("‼️ 이미지 무결성 오류 발생:")
        for err in integrity_errors:
            print("  -", err)
    else:
        print("✅ 모든 이미지가 정상적으로 로드됩니다.")

def main():
    print("===== Train 데이터셋 검증 =====")
    check_integrity(TRAIN_IMAGES, TRAIN_LABELS)
    print("\n===== Validation 데이터셋 검증 =====")
    check_integrity(VAL_IMAGES, VAL_LABELS)
    
if __name__ == "__main__":
    main()
