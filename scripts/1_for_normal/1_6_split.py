import os
import shutil
from glob import glob

# 소스 폴더 경로 설정
train_source = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/5_1_yolo_augmented_output"
val_txt_source = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/4_2_val_txt"
val_img_source = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/4_4_val_image"

# 대상 폴더 경로 설정
train_labels_dest = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/train/labels"
train_images_dest = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/train/images"
val_labels_dest = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/val/labels"
val_images_dest = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/val/images"

# 대상 폴더가 없으면 생성한다.
for folder in [train_labels_dest, train_images_dest, val_labels_dest, val_images_dest]:
    os.makedirs(folder, exist_ok=True)
    print(f"생성된 폴더: {folder}")

# 5_1_yolo_augmented_output 내의 txt 파일을 train/labels로 복사한다.
train_txt_files = glob(os.path.join(train_source, "*.txt"))
for file_path in train_txt_files:
    dest_path = os.path.join(train_labels_dest, os.path.basename(file_path))
    print(f"복사: {file_path} -> {dest_path}")
    shutil.copy2(file_path, dest_path)

# 5_1_yolo_augmented_output 내의 jpg 파일을 train/images로 복사한다.
train_jpg_files = glob(os.path.join(train_source, "*.jpg"))
for file_path in train_jpg_files:
    dest_path = os.path.join(train_images_dest, os.path.basename(file_path))
    print(f"복사: {file_path} -> {dest_path}")
    shutil.copy2(file_path, dest_path)

# 4_2_val_txt 내의 txt 파일을 val/labels로 복사한다.
val_txt_files = glob(os.path.join(val_txt_source, "*.txt"))
for file_path in val_txt_files:
    dest_path = os.path.join(val_labels_dest, os.path.basename(file_path))
    print(f"복사: {file_path} -> {dest_path}")
    shutil.copy2(file_path, dest_path)

# 4_4_val_image 내의 jpg 파일을 val/images로 복사한다.
val_jpg_files = glob(os.path.join(val_img_source, "*.jpg"))
for file_path in val_jpg_files:
    dest_path = os.path.join(val_images_dest, os.path.basename(file_path))
    print(f"복사: {file_path} -> {dest_path}")
    shutil.copy2(file_path, dest_path)

print("파일 복사 완료")
