#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 1. YOLO OBB 라벨 -> COCO GT 변환 (픽셀 좌표 사용)
def convert_yolo_obb_to_coco(labels_dir, coco_output_file, image_dir, class_names, img_width, img_height):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i+1, "name": name} for i, name in enumerate(class_names)]
    }
    annotation_id = 1
    image_id = 1

    for label_file in sorted(os.listdir(labels_dir)):
        if not label_file.endswith(".txt"):
            continue
        img_name = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"⚠ Warning: 이미지 {img_path} 없음. 건너뜀.")
            continue
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": img_width,
            "height": img_height
        })
        with open(os.path.join(labels_dir, label_file), "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0]) + 1
            coords = list(map(float, parts[1:]))

            if len(coords) != 8:
                print(f"⚠ Warning: {label_file} 라벨 데이터 오류 (좌표 개수 불일치). 건너뜀.")
                continue

            x1, y1, x2, y2, x3, y3, x4, y4 = coords
            x1, y1 = x1 * img_width, y1 * img_height
            x2, y2 = x2 * img_width, y2 * img_height
            x3, y3 = x3 * img_width, y3 * img_height
            x4, y4 = x4 * img_width, y4 * img_height

            x_min = min(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            x_max = max(x1, x2, x3, x4)
            y_max = max(y1, y2, y3, y4)

            width = x_max - x_min
            height = y_max - y_min

            bbox = [x_min, y_min, width, height]
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": bbox,
                "area": width * height,
                "iscrowd": 0
            })
            annotation_id += 1
        image_id += 1

    with open(coco_output_file, "w") as f:
        json.dump(coco_data, f, indent=4)
    print(f"✅ GT COCO JSON 변환 완료: {coco_output_file}")

# 2. YOLO 예측(rbox) -> COCO 예측(픽셀 좌표) 변환
def convert_yolo_pred_to_coco(yolo_pred_file, coco_output_file, img_width, img_height):
    with open(yolo_pred_file, "r") as f:
        yolo_preds = json.load(f)
    coco_results = []
    for pred in yolo_preds:
        # pred["image_id"]가 이미지 파일 이름이라고 가정
        file_name = pred["image_id"]
        category_id = pred["category_id"] + 1
        score = pred["score"]
        # rbox = [x_center, y_center, w, h, theta]
        rbox = pred["rbox"]
        x_center, y_center, w, h, _ = rbox
        x = x_center - (w / 2)
        y = y_center - (h / 2)
        # 픽셀 단위로 변환
        x *= img_width
        y *= img_height
        w *= img_width
        h *= img_height
        coco_results.append({
            "image_id": file_name,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "score": score
        })
    with open(coco_output_file, "w") as f:
        json.dump(coco_results, f, indent=4)
    print(f"✅ 예측 COCO JSON 변환 완료: {coco_output_file}")

# 3. image_id를 숫자로 매핑 (GT에 맞춤)
def fix_image_id(gt_file, pred_file, output_file):
    with open(gt_file, "r") as f:
        gt_data = json.load(f)
    # 확장자 제외한 파일이름 -> image_id
    image_id_map = {}
    for img in gt_data["images"]:
        file_stem = os.path.splitext(img["file_name"])[0]
        image_id_map[file_stem] = img["id"]
    with open(pred_file, "r") as f:
        pred_data = json.load(f)
    fixed_preds = []
    for pred in pred_data:
        file_stem = os.path.splitext(str(pred["image_id"]))[0]
        if file_stem in image_id_map:
            new_pred = dict(pred)
            new_pred["image_id"] = image_id_map[file_stem]
            fixed_preds.append(new_pred)
        else:
            print(f"⚠ Warning: {file_stem}이(가) GT에 없음. 제거.")
    with open(output_file, "w") as f:
        json.dump(fixed_preds, f, indent=4)
    print(f"✅ image_id 매핑 완료: {output_file}")

# 4. category_id를 GT에 맞춰 통일 (단순화 버전)
def fix_category_id(gt_file, pred_file, output_file):
    with open(gt_file, "r") as f:
        gt_data = json.load(f)
    if not gt_data["categories"]:
        print("⚠ Warning: GT에 categories 정보가 없음.")
        return
    # 보통 첫 번째 카테고리 id를 그대로 사용
    gt_category_id = gt_data["categories"][0]["id"]
    with open(pred_file, "r") as f:
        pred_data = json.load(f)
    for pred in pred_data:
        pred["category_id"] = gt_category_id
    with open(output_file, "w") as f:
        json.dump(pred_data, f, indent=4)
    print(f"✅ category_id 통일 완료: {output_file}")

# 5. COCO AP/AR 평가
def coco_evaluation(gt_file, dt_file):
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(dt_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

# 6. 메인 실행부
if __name__ == "__main__":
    # 경로와 파라미터는 필요에 맞게 수정한다.
    yolo_labels_dir = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset/labels/val"
    image_dir = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset/images/val"
    class_names = ["component"] 
    img_width = 3904
    img_height = 3904

    # GT 출력
    gt_file_pixel = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset/ground_truth_pixel.json"
    convert_yolo_obb_to_coco(yolo_labels_dir, gt_file_pixel, image_dir, class_names, img_width, img_height)

    # 예측 변환
    yolo_pred_file = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/scripts/runs/obb/val/predictions.json"
    coco_pred_file_raw = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/outputs_for_exper/run/coco_predictions_pixel_raw.json"
    convert_yolo_pred_to_coco(yolo_pred_file, coco_pred_file_raw, img_width, img_height)

    # image_id & category_id 수정
    coco_pred_file_fixed = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/outputs_for_exper/run/coco_predictions_pixel_fixed.json"
    fix_image_id(gt_file_pixel, coco_pred_file_raw, coco_pred_file_fixed)

    coco_pred_file_final = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/outputs_for_exper/run/coco_predictions_pixel_final.json"
    fix_category_id(gt_file_pixel, coco_pred_file_fixed, coco_pred_file_final)

    # 최종 평가
    results = coco_evaluation(gt_file_pixel, coco_pred_file_final)
    print("COCO Evaluation Results:", results)
