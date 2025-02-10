# scripts/custom_obb_validator.py
import torch
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils.metrics import ap_per_class

class SizeAwareOBBValidator(BaseValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size_categories = {
            "small": (0, 32**2),
            "medium": (32**2, 96**2),
            "large": (96**2, float("inf"))
        }

    def _calculate_obb_area(self, boxes):
        x = boxes[..., ::2]  # x1, x2, x3, x4
        y = boxes[..., 1::2]  # y1, y2, y3, y4
        return 0.5 * torch.abs(
            (x[..., 0] * y[..., 1] + x[..., 1] * y[..., 2] + x[..., 2] * y[..., 3] + x[..., 3] * y[..., 0]) -
            (x[..., 1] * y[..., 0] + x[..., 2] * y[..., 1] + x[..., 3] * y[..., 2] + x[..., 0] * y[..., 3])
        )

    def get_metrics(self):
        metrics = super().get_metrics()
        for size_name, (min_area, max_area) in self.size_categories.items():
            pred_areas = self._calculate_obb_area(self.pred[:, :8])
            gt_areas = self._calculate_obb_area(self.gt[:, :8])
            mask = (gt_areas >= min_area) & (gt_areas < max_area)
            _, _, _, map50, map = ap_per_class(self.tp, self.conf, self.pred_cls, self.target_cls, plot=False, names=self.names)[:5]
            metrics[f'mAP50_{size_name}'] = map50[mask].mean()
            metrics[f'mAP_{size_name}'] = map[mask].mean()
        return metrics