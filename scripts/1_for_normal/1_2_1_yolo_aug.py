import os
import cv2
import albumentations as A
from glob import glob

print("Initializing Data Augmentation...")

# 원본 이미지와 라벨, 출력 디렉토리 경로를 설정한다.
input_images = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/1_1_800images"
input_labels = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/4_1_train_txt"
output_dir = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/yolo_augmented_output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# 다양한 증강 기법을 적용한다.
def get_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15,
                           interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomSizedBBoxSafeCrop(height=640, width=640, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category']))

# 한 이미지에 대해 증강을 적용하는 함수이다.
def augment_data(image_path, label_path, output_dir, num_augmentations=10):
    print(f"Processing: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    print(f"Loaded image: {image_path} with shape {image.shape}")
    
    with open(label_path, "r") as f:
        labels = f.readlines()
    
    bboxes = []
    categories = []
    for label in labels:
        data = label.strip().split()
        if len(data) < 5:
            print(f"Warning: Skipping invalid label {label} in {label_path}")
            continue
        class_id = int(data[0])
        bbox = list(map(float, data[1:]))
        bboxes.append(bbox)
        categories.append(class_id)
        
    if not bboxes:
        print(f"Warning: No valid bounding boxes found in {label_path}")
        return
        
    print(f"Applying augmentations to {image_path}")
    successful = 0
    attempts = 0
    max_attempts = num_augmentations * 3  # 실패 시 재시도 최대 횟수 설정
    while successful < num_augmentations and attempts < max_attempts:
        attempts += 1
        transform = get_augmentation()
        try:
            augmented = transform(image=image, bboxes=bboxes, category=categories)
        except ValueError as e:
            print(f"Warning: Augmentation failed with error: {e}. Retrying...")
            continue
        
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['category']
        
        aug_image_path = os.path.join(output_dir, f"aug_{successful+1}_" + os.path.basename(image_path))
        cv2.imwrite(aug_image_path, aug_image)
        
        aug_label_path = os.path.join(output_dir, f"aug_{successful+1}_" + os.path.basename(label_path))
        with open(aug_label_path, "w") as f:
            for bbox, cls in zip(aug_bboxes, aug_labels):
                f.write(f"{cls} " + " ".join(map(str, bbox)) + "\n")
        print(f"Saved: {aug_image_path}")
        successful += 1
    if successful < num_augmentations:
        print(f"Warning: Only {successful} augmentations were generated for {image_path} after {attempts} attempts.")

# 원본 이미지 목록을 가져와서 각 이미지에 대해 증강을 적용한다.
image_paths = glob(os.path.join(input_images, "*.jpg"))
print(f"Found {len(image_paths)} images")

for img_path in image_paths:
    # 이미지 파일의 기본 이름을 추출하여 대응되는 라벨 파일을 구성한다.
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(input_labels, base_name + ".txt")
    if os.path.exists(label_path):
        augment_data(img_path, label_path, output_dir, num_augmentations=10)
    else:
        print(f"Warning: Label file not found for {img_path}")
