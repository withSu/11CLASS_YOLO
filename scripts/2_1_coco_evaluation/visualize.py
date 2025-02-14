# 파일명: /home/a/A_2024_selfcode/CLASS-PCB_Yolo/scripts/visualize_and_print_summary.py

import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_iou(box1, box2):
    # box 형식: [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def draw_boxes(image, boxes, color, thickness=1):
    for box_info in boxes:
        x, y, w, h = map(int, box_info["bbox"])
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
    return image

def overlay_text(image, lines, start_pos=(5,20), font_scale=0.3, color=(255,255,255), thickness=1, line_height=15):
    x, y = start_pos
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y += line_height
    return image

def evaluate_coco(gt_json_path, pred_json_path):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    # mAP IoU=0.50:0.95 평가
    coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

def visualize_and_print_summary(gt_json_path, pred_json_path, images_dir, output_dir, iou_threshold=0.5):
    gt_data = load_json(gt_json_path)
    pred_data = load_json(pred_json_path)
    
    # 이미지 ID별로 GT와 예측 박스를 분류한다.
    gt_boxes = {}
    for ann in gt_data["annotations"]:
        image_id = ann["image_id"]
        gt_boxes.setdefault(image_id, []).append(ann)
    
    pred_boxes = {}
    for ann in pred_data:
        image_id = ann["image_id"]
        pred_boxes.setdefault(image_id, []).append(ann)
    
    # 이미지 ID와 파일명을 매핑한다.
    image_id_to_file = {}
    for img in gt_data["images"]:
        image_id_to_file[img["id"]] = img["file_name"]
    
    total_gt_boxes = 0
    total_missed_boxes = 0
    missed_classes = {}
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 각 이미지별 시각화 및 이미지 내 요약 오버레이
    for image_id, file_name in image_id_to_file.items():
        image_path = os.path.join(images_dir, file_name)
        if not os.path.exists(image_path):
            print("이미지 파일을 찾지 못함:", image_path)
            continue
        image = cv2.imread(image_path)
        if image is None:
            print("이미지를 불러오지 못함:", image_path)
            continue
        
        gt_list = gt_boxes.get(image_id, [])
        pred_list = pred_boxes.get(image_id, [])
        
        # 이미지에 GT 박스(초록색)와 예측 박스(빨간색)를 그림
        image = draw_boxes(image, gt_list, (0, 255, 0), thickness=1)
        image = draw_boxes(image, pred_list, (0, 0, 255), thickness=1)
        
        # 각 GT 박스에 대해 최고 IoU가 iou_threshold 미만이면 미스된 것으로 간주
        missed_count = 0
        for gt in gt_list:
            max_iou = 0.0
            for pred in pred_list:
                iou = compute_iou(gt["bbox"], pred["bbox"])
                if iou > max_iou:
                    max_iou = iou
            if max_iou < iou_threshold:
                missed_count += 1
                cat_id = gt["category_id"]
                # gt_data["categories"]에 기록된 category_id는 0부터 시작한다.
                cat_name = next((cat["name"] for cat in gt_data["categories"] if cat["id"] == cat_id), str(cat_id))
                missed_classes[cat_name] = missed_classes.get(cat_name, 0) + 1
        
        total_gt_boxes += len(gt_list)
        total_missed_boxes += missed_count
        
        # 이미지 내 오버레이 텍스트 (per-image summary)
        summary_lines = [
            f"Image: {file_name}",
            f"Total GT Boxes: {len(gt_list)}",
            f"Missed Boxes: {missed_count}"
        ]
        image = overlay_text(image, summary_lines, start_pos=(5,15), font_scale=0.3, line_height=15)
        
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, image)
    
    # 전체 요약 출력 (터미널)
    print("==== Overall Summary ====")
    print(f"Total GT Boxes: {total_gt_boxes}")
    print(f"Total Missed Boxes: {total_missed_boxes}")
    if total_gt_boxes > 0:
        print(f"Missed Percentage: {(total_missed_boxes / total_gt_boxes) * 100:.2f}%\n")
    else:
        print("Missed Percentage: N/A\n")
    
    print("==== Missed Classes ====")
    for label, count in missed_classes.items():
        print(f"  {label}: {count} boxes")
    
    # COCO Evaluation (터미널 출력)
    stats = evaluate_coco(gt_json_path, pred_json_path)
    print("\n==== COCO Evaluation ====")
    print(f"mAP (IoU=0.50:0.95): {stats[0]:.3f}")
    print(f"mAP (IoU=0.50): {stats[1]:.3f}")

if __name__ == "__main__":
    gt_json = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/coco_gt.json"
    pred_json = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/runs/detect/val4/predictions_fixed.json"
    images_dir = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/val/images"
    output_dir = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/runs/detect/val4/visualization"
    visualize_and_print_summary(gt_json, pred_json, images_dir, output_dir, iou_threshold=0.5)
