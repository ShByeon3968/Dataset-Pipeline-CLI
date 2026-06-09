from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class Detection:
    x: float          # top-left x (0~1 normalized)
    y: float          # top-left y
    w: float          # width
    h: float          # height
    class_id: int
    confidence: float


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Pure-numpy NMS. boxes_xyxy: [N,4] xyxy normalized."""
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


class BaseDetector(ABC):

    @abstractmethod
    def preprocess(self, bgr: np.ndarray, input_w: int, input_h: int) -> np.ndarray:
        """BGR ndarray → 모델 입력 텐서 [1,3,H,W] float32"""
        ...

    @abstractmethod
    def postprocess(
        self,
        outputs: list[np.ndarray],
        orig_w: int,
        orig_h: int,
        input_w: int,
        input_h: int,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[Detection]:
        ...

    def _letterbox(self, bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        import cv2
        h, w = bgr.shape[:2]
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (nw, nh))
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        return canvas
