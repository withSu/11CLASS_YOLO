import json
import os
import cv2
import matplotlib.pyplot as plt
import random

def calculate_iou(boxA, boxB):
    """
    두 박스(boxA, boxB)는 (x, y, w, h)의 픽셀 좌표이다.
    두 박스의 IoU(교집합 비율)를 계산한다.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0:
        return 0.0

    iou = interArea / unionArea
    return iou

def get_size_category(w, h):
    """
    COCO에서 small, medium, large를 면적 기준으로 구분한다.
    small: area < 32*32
    medium: 32*32 <= area < 96*96
    large: area >= 96*96
    """
    area = w * h
    if area < 32 * 32:
        return "small"
    elif area < 96 * 96:
        return "medium"
    else:
        return "large"

def visualize_and_iou(gt_file, pred_file, image_dir, num_samples=5):
    """
    GT와 예측을 로드해, 임의의 이미지를 뽑아 바운딩 박스를 시각화한다.
    정규화된 bbox는 이미지 폭, 높이를 곱해 픽셀 단위로 변환한다.
    IoU를 간단히 계산하고, 평균값을 표시한다.
    """
    with open(gt_file, "r") as f:
        gt_data = json.load(f)

    with open(pred_file, "r") as f:
        pred_data = json.load(f)

    # GT images의 {image_id: file_name} 매핑
    image_id_to_file = {}
    for img_info in gt_data["images"]:
        image_id_to_file[img_info["id"]] = img_info["file_name"]

    # 랜덤으로 보여줄 이미지 ID 샘플
    all_ids = list(image_id_to_file.keys())
    if not all_ids:
        print("이미지가 없습니다.")
        return
    sample_ids = random.sample(all_ids, min(num_samples, len(all_ids)))

    for image_id in sample_ids:
        img_file = image_id_to_file[image_id]
        img_path = os.path.join(image_dir, img_file)

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠ {img_path} 읽기 실패. 스킵한다.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = img.shape

        # GT bboxes
        gt_bboxes = [ann["bbox"] for ann in gt_data["annotations"] if ann["image_id"] == image_id]
        # 예측 bboxes
        pred_bboxes = [dt["bbox"] for dt in pred_data if dt["image_id"] == image_id]

        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        # 파란색: GT 바운딩박스
        for bbox in gt_bboxes:
            gx, gy, gw, gh = bbox
            px = gx
            py = gy
            pw = gw
            ph = gh

            # 만약 bbox가 정규화되어 있으면, 아래처럼 w_img, h_img 곱하기
            # px, py, pw, ph = gx*w_img, gy*h_img, gw*w_img, gh*h_img

            plt.gca().add_patch(
                plt.Rectangle((px, py), pw, ph, edgecolor='blue', linewidth=1, fill=False)
            )

        # 빨간색: 예측 바운딩박스
        for bbox in pred_bboxes:
            px, py, pw, ph = bbox
            # 만약 정규화면
            # px, py, pw, ph = px*w_img, py*h_img, pw*w_img, ph*h_img

            size_cat = get_size_category(pw, ph)
            plt.gca().add_patch(
                plt.Rectangle((px, py), pw, ph, edgecolor='red', linewidth=1, fill=False)
            )
            plt.text(px+1, py+10, f"pred: {size_cat}", color="red", fontsize=7)

        # IoU 간단 계산
        gt_pixels = []
        for bbox in gt_bboxes:
            gx, gy, gw, gh = bbox
            # 정규화 되어 있다면 곱해야 함
            # gx, gy, gw, gh = gx*w_img, gy*h_img, gw*w_img, gh*h_img
            gt_pixels.append((gx, gy, gw, gh))

        pred_pixels = []
        for bbox in pred_bboxes:
            px, py, pw, ph = bbox
            # 정규화 되어 있다면 곱해야 함
            # px, py, pw, ph = px*w_img, py*h_img, pw*w_img, ph*h_img
            pred_pixels.append((px, py, pw, ph))

        iou_scores = []
        for gbox in gt_pixels:
            for pbox in pred_pixels:
                iou_val = calculate_iou(gbox, pbox)
                iou_scores.append(iou_val)

        avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
        plt.title(f"{img_file} - 평균 IoU: {avg_iou:.3f}", fontsize=11)
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    # 사용자가 원하는 경로
    gt_file = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/ground_truth.json"
    pred_file = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/runs/detect/val4/coco_predictions.json"
    image_dir = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/val/images"

    # 시각화 함수 실행
    visualize_and_iou(gt_file, pred_file, image_dir, num_samples=5)