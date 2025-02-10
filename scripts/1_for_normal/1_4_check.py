# input: 1_2_800images, 4_800labels
# output: ✅ 모든 이미지와 라벨 파일이 정확히 일치합니다.
import os

# 경로 설정
IMAGES_DIR = '/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset/1_1_800images'
LABELS_DIR = '/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset/4_800size_txt_labels'

# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

def check_image_label_matching():
    # 이미지 파일 이름 (확장자 제거)
    image_files = {os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR) if os.path.splitext(f)[1] in IMAGE_EXTENSIONS}
    
    # 라벨 파일 이름 (.txt 제거)
    label_files = {os.path.splitext(f)[0] for f in os.listdir(LABELS_DIR) if f.endswith('.txt')}
    
    # 일치하지 않는 이미지 및 라벨 확인
    unmatched_images = image_files - label_files
    unmatched_labels = label_files - image_files
    
    print(f"🔍 총 이미지 파일 수: {len(image_files)}")
    print(f"🔍 총 라벨 파일 수: {len(label_files)}")
    
    if unmatched_images:
        print(f"⚠️ 매칭되지 않은 이미지 파일: {len(unmatched_images)}")
        for img in unmatched_images:
            print(f" - {img}")
    
    if unmatched_labels:
        print(f"⚠️ 매칭되지 않은 라벨 파일: {len(unmatched_labels)}")
        for lbl in unmatched_labels:
            print(f" - {lbl}")
    
    if not unmatched_images and not unmatched_labels:
        print("✅ 모든 이미지와 라벨 파일이 정확히 일치합니다.")

if __name__ == '__main__':
    check_image_label_matching()
