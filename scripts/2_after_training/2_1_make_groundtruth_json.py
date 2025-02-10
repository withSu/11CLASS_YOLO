import os
import json
from PIL import Image

def create_ground_truth_json(
    image_dir = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset/val/images",
    label_dir = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset/val/labels",
    output_json = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset/ground_truth.json"
):
    """
    YOLO 라벨(txt) 파일을 COCO 형식의 ground_truth.json으로 변환한다.
    이미지 해상도가 800×800이라고 가정하고, 라벨에 있는 x_center, y_center, w, h는 (0~1) 정규화된 좌표라고 가정한다.
    """

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    images_info = []
    annotations = []
    categories = {}
    annotation_id = 1
    image_id = 1

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        with Image.open(img_path) as img:
            width, height = img.size  # 보통 800×800

        images_info.append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": img_file
        })

        base_name, _ = os.path.splitext(img_file)
        label_file = os.path.join(label_dir, base_name + ".txt")
        if not os.path.exists(label_file):
            image_id += 1
            continue

        with open(label_file, "r") as lf:
            lines = lf.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            x_center_abs = x_center * width
            y_center_abs = y_center * height
            w_abs = w * width
            h_abs = h * height

            x_min = x_center_abs - (w_abs / 2.0)
            y_min = y_center_abs - (h_abs / 2.0)

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x_min, y_min, w_abs, h_abs],
                "area": w_abs * h_abs,
                "iscrowd": 0
            })
            annotation_id += 1

            if class_id not in categories:
                categories[class_id] = f"class_{class_id}"

        image_id += 1

    categories_info = []
    for cat_id, cat_name in sorted(categories.items()):
        categories_info.append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "none"
        })

    coco_format = {
        "images": images_info,
        "annotations": annotations,
        "categories": categories_info
    }

    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)

    print("✅ COCO 형식 ground_truth.json 생성 완료:", output_json)

if __name__ == "__main__":
    create_ground_truth_json()
