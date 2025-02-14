import os
import glob
import json
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def yolo_to_coco(dataset_dir, output_json):
    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")
    category_names = [
        "Chip", "CSolder", "2sideIC", "SOD", "Circle",
        "4sideIC", "Tantalum", "BGA", "MELF", "Crystal", "Array"
    ]
    coco_data = {
        "info": {
            "description": "YOLO -> COCO 변환 예시",
            "version": "1.0"
        },
        "images": [],
        "annotations": [],
        "categories": []
    }

    # category_id가 0부터 시작하도록 설정
    for i, cat_name in enumerate(category_names):
        coco_data["categories"].append({
            "id": i,
            "name": cat_name
        })

    annotation_id = 1
    image_id = 1
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_paths += glob.glob(os.path.join(image_dir, ext))
    image_paths.sort()

    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        stem, _ = os.path.splitext(file_name)
        label_path = os.path.join(label_dir, stem + ".txt")

        with Image.open(img_path) as img:
            w_img, h_img = img.size

        coco_data["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": w_img,
            "height": h_img
        })

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    x_min = (x_center - w_norm/2) * w_img
                    y_min = (y_center - h_norm/2) * h_img
                    bbox_w = w_norm * w_img
                    bbox_h = h_norm * h_img

                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x_min, y_min, bbox_w, bbox_h],
                        "area": bbox_w * bbox_h,
                        "iscrowd": 0
                    })
                    annotation_id += 1

        image_id += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    print("YOLO 라벨 → COCO 변환 완료:", output_json)

def fix_predictions(dt_json_path, gt_json_path, is_normalized=False):
    coco_gt = COCO(gt_json_path)

    # Ground truth의 이미지 파일명을 소문자로 변환하여 매핑
    filename_to_id = {}
    for img_id, info in coco_gt.imgs.items():
        fn = info["file_name"].lower()
        stem = os.path.splitext(fn)[0]
        filename_to_id[fn] = img_id
        filename_to_id[stem] = img_id

    with open(dt_json_path, "r") as f:
        dt_data = json.load(f)

    fixed = []
    possible_exts = [".jpg", ".jpeg", ".png"]

    for pred in dt_data:
        file_or_id = pred["image_id"]
        cat_id = pred["category_id"]
        bbox = pred["bbox"]
        score = pred.get("score", 1.0)

        if isinstance(file_or_id, int):
            new_id = file_or_id
        else:
            file_or_id_lower = file_or_id.lower()
            stem, ext = os.path.splitext(file_or_id_lower)
            candidates = [file_or_id_lower] + [stem + e for e in possible_exts if not ext]
            new_id = next((filename_to_id[candi] for candi in candidates if candi in filename_to_id), None)

            if new_id is None:
                print("⚠ 매칭 실패:", file_or_id)
                continue

        if is_normalized:
            img_info = coco_gt.imgs[new_id]
            w_img, h_img = img_info["width"], img_info["height"]
            x_cen, y_cen, w_norm, h_norm = bbox
            x_min = (x_cen - w_norm/2) * w_img
            y_min = (y_cen - h_norm/2) * h_img
            w_box = w_norm * w_img
            h_box = h_norm * h_img
            bbox = [x_min, y_min, w_box, h_box]

        fixed.append({
            "image_id": new_id,
            "category_id": cat_id,
            "bbox": bbox,
            "score": score
        })

    dt_fixed_path = dt_json_path.replace(".json", "_fixed.json")
    with open(dt_fixed_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, indent=2)

    print("예측 JSON 수정 완료:", dt_fixed_path)
    print("총 예측 수:", len(dt_data), "/ 유효 매칭 수:", len(fixed))
    return dt_fixed_path

def coco_evaluation(gt_json_path, dt_json_path):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(dt_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # DETR과 동일하게 IoU 임계값 설정
    coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats

if __name__ == "__main__":
    dataset_dir = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/val"
    gt_json = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/coco_gt.json"
    yolo_to_coco(dataset_dir, gt_json)

    dt_json = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/runs/detect/val4/predictions.json"
    dt_fixed = fix_predictions(dt_json, gt_json, is_normalized=False)

    results = coco_evaluation(gt_json, dt_fixed)
    print("COCO Evaluation Results:", results)
