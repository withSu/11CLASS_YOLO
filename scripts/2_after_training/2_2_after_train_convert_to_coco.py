import json
import os

def convert_yolo_to_coco(yolo_pred_file, coco_output_file, gt_file):
    """
    일반 YOLO 예측 결과를 COCO 평가 JSON 형식으로 변환한다.
    (1) GT 파일에서 file_name(확장자 제거) → 정수 image_id 매핑을 만든다.
    (2) 예측(JSON)의 "bbox"는 이미 [x_min, y_min, w, h] 픽셀 좌표라고 가정한다.
    (3) "image_id"(문자열)를 GT의 정수 id로 치환한다.
    (4) GT가 단일 클래스라면, 예측 category_id도 GT category_id(예: 0)로 강제 변경한다.
    (5) 최종 JSON을 coco_output_file에 저장한다.
    """

    # (A) GT 파일 로드
    if not os.path.exists(gt_file):
        print(f"❌ GT 파일이 존재하지 않습니다: {gt_file}")
        return

    with open(gt_file, "r") as f:
        gt_data = json.load(f)

    # GT가 단일 카테고리라고 가정 (categories[0]["id"]만 사용)
    if len(gt_data["categories"]) == 1:
        gt_cat_id = gt_data["categories"][0]["id"]  # 보통 0
    else:
        # 여러 클래스가 있다면, 별도 매핑 필요
        gt_cat_id = None

    # file_name(확장자 제거) → 정수 image_id 매핑
    filename2id = {}
    for img_info in gt_data["images"]:
        full_name = img_info["file_name"]
        base_name = os.path.splitext(full_name)[0]
        filename2id[base_name] = img_info["id"]

    # (B) 예측 파일 로드
    if not os.path.exists(yolo_pred_file):
        print(f"❌ 예측 파일이 존재하지 않습니다: {yolo_pred_file}")
        return

    with open(yolo_pred_file, "r") as f:
        yolo_preds = json.load(f)

    coco_results = []
    for pred in yolo_preds:
        image_id_str = str(pred["image_id"])
        if image_id_str not in filename2id:
            # GT에 없는 파일명이면 스킵
            continue

        image_id_int = filename2id[image_id_str]
        
        # (C) category_id
        category_id = pred["category_id"]
        # 만약 단일 클래스라면, GT 카테고리 id로 강제
        if gt_cat_id is not None:
            category_id = gt_cat_id

        score = pred["score"]

        # (D) bbox
        x, y, w, h = pred["bbox"]

        coco_results.append({
            "image_id": image_id_int,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "score": score
        })

    # (E) 결과 저장
    with open(coco_output_file, "w") as f:
        json.dump(coco_results, f, indent=4)

    print(f"✅ COCO 평가용 JSON 변환 완료: {coco_output_file}")

if __name__ == "__main__":
    gt_file = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset/ground_truth.json"
    yolo_pred_file = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/outputs_800yolo/run3/predictions.json"
    coco_output_file = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/outputs_800yolo/run3/coco_predictions.json"

    convert_yolo_to_coco(yolo_pred_file, coco_output_file, gt_file)
