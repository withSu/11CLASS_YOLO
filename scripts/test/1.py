from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# COCO 데이터셋 로드
coco_gt = COCO("/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/coco_gt.json")  # COCO 검증 데이터셋 (Ground Truth)
coco_dt = coco_gt.loadRes("/home/a/A_2024_selfcode/CLASS-PCB_Yolo/runs/detect/val4/predictions.json")  # YOLO의 예측 결과 JSON

# 평가 객체 생성
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")  # bbox 평가
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()  # 결과 출력
