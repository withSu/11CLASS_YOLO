# 파란색이 정답값 
# 빨간색이 예측값 

import json
import os
import cv2
import matplotlib.pyplot as plt
import random

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

# 데이터 로드
gt_file = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/ground_truth.json"
pred_file = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/outputs_for_exper/run/coco_predictions_final_normalized.json"
image_dir = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/images/val"  

with open(gt_file, "r") as f:
    gt_data = json.load(f)

with open(pred_file, "r") as f:
    pred_data = json.load(f)

# GT에서 image_id와 file_name 매핑
image_id_to_filename = {img["id"]: img["file_name"] for img in gt_data["images"]}

# 랜덤한 이미지 선택 (최대 5개)
sample_images = random.sample(list(image_id_to_filename.keys()), min(5, len(image_id_to_filename)))

for image_id in sample_images:
    image_name = image_id_to_filename[image_id]
    image_path = os.path.join(image_dir, image_name)

    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠ Warning: {image_path} 로드 실패. 스킵합니다.")
        continue

    # OpenCV -> matplotlib 순서 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img.shape

    # GT와 Prediction BBox 찾기
    gt_bboxes = [ann["bbox"] for ann in gt_data["annotations"] if ann["image_id"] == image_id]
    pred_bboxes = [pred["bbox"] for pred in pred_data if pred["image_id"] == image_id]

    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    # GT (파란색) -> 픽셀 변환 후 그리기
    for bbox in gt_bboxes:
        x, y, w, h = bbox
        # 정규화된 좌표를 픽셀 단위로 변환
        px = x * img_w
        py = y * img_h
        pw = w * img_w
        ph = h * img_h

        plt.gca().add_patch(
            plt.Rectangle((px, py), pw, ph, edgecolor='blue', linewidth=2, fill=False)
        )

    # Prediction (빨간색) -> 픽셀 변환 후 그리기
    for bbox in pred_bboxes:
        x, y, w, h = bbox
        px = x * img_w
        py = y * img_h
        pw = w * img_w
        ph = h * img_h

        plt.gca().add_patch(
            plt.Rectangle((px, py), pw, ph, edgecolor='red', linewidth=2, fill=False)
        )

    # IoU 계산 (IoU는 픽셀 단위로 해석하는 게 일반적)
    # GT도 픽셀 단위 변환한 뒤 IoU 계산하는 편이 좋다.
    gt_bboxes_pixel = [(box[0]*img_w, box[1]*img_h, box[2]*img_w, box[3]*img_h) for box in gt_bboxes]
    pred_bboxes_pixel = [(box[0]*img_w, box[1]*img_h, box[2]*img_w, box[3]*img_h) for box in pred_bboxes]
    iou_scores = [calculate_iou(gtb, pb) for gtb in gt_bboxes_pixel for pb in pred_bboxes_pixel]
    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0

    plt.title(f"{image_name}\nAvg IoU: {avg_iou:.2f}")
    # 범례 중복 추가 방지를 위해 별도 legend는 생략하거나 필요시 그리기
    plt.axis("off")
    plt.show()
