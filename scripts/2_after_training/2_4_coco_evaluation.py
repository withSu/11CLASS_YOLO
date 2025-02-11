import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def coco_evaluation(gt_file, dt_file):
    """
    COCO AP/AR 평가를 COCO 공식 기준(IoU 0.50~0.95, 0.05 간격)으로 수행한다.
    gt_file과 dt_file은 모두 COCO 형식을 따르는 JSON 경로이다.
    bbox는 [x_min, y_min, width, height] 픽셀 좌표라고 가정한다.
    """
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(dt_file)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats

if __name__ == "__main__":
    gt_path = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/ground_truth.json"
    dt_path = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/runs/detect/val/coco_predictions.json"

    results = coco_evaluation(gt_path, dt_path)
    print("COCO Evaluation Results:", results)
