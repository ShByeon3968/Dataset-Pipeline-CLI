import numpy as np
from .base import BaseDetector, Detection


class RFDETRDetector(BaseDetector):
    """
    RF-DETR / RT-DETR ONNX 출력 형식 두 가지를 자동 감지:

    [형식 A]  orig_target_sizes 없이 export된 경우
      outputs[0] = pred_logits [1, N, nc]  (raw logits)
      outputs[1] = pred_boxes  [1, N, 4]   cxcywh normalized

    [형식 B]  orig_target_sizes 포함 export된 경우 (일반적인 roboflow RF-DETR)
      outputs[0] = labels  [1, N]    int64 class index
      outputs[1] = boxes   [1, N, 4] xyxy pixel coords (orig image space)
      outputs[2] = scores  [1, N]    float confidence
    """

    def preprocess(self, bgr: np.ndarray, input_w: int, input_h: int) -> np.ndarray:
        import cv2
        img = cv2.resize(bgr, (input_w, input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

    def postprocess(self, outputs, orig_w, orig_h, input_w, input_h, conf_threshold, iou_threshold):
        first = outputs[0][0]  # squeeze batch dim

        # ── 형식 판별: first.ndim == 1 이면 형식 B (labels), 2 이면 형식 A (pred_logits)
        if first.ndim == 1:
            return self._postprocess_b(outputs, orig_w, orig_h, conf_threshold)
        else:
            return self._postprocess_a(outputs, orig_w, orig_h, conf_threshold)

    # ── 형식 A: pred_logits + pred_boxes cxcywh normalized ──────────────────
    def _postprocess_a(self, outputs, orig_w, orig_h, conf_threshold):
        logits = outputs[0][0]   # [N, nc]
        boxes  = outputs[1][0]   # [N, 4]  cxcywh normalized

        scores      = 1.0 / (1.0 + np.exp(-logits))   # sigmoid
        class_ids   = scores.argmax(axis=1)
        confidences = scores[np.arange(len(class_ids)), class_ids]

        mask = confidences >= conf_threshold
        boxes, class_ids, confidences = boxes[mask], class_ids[mask], confidences[mask]

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return [
            Detection(
                x=float(cx[i] - bw[i] / 2),
                y=float(cy[i] - bh[i] / 2),
                w=float(bw[i]),
                h=float(bh[i]),
                class_id=int(class_ids[i]),
                confidence=float(confidences[i]),
            )
            for i in range(len(class_ids))
        ]

    # ── 형식 B: labels + boxes(xyxy pixel) + scores ─────────────────────────
    def _postprocess_b(self, outputs, orig_w, orig_h, conf_threshold):
        class_ids   = outputs[0][0].astype(int)   # [N]
        boxes       = outputs[1][0]                # [N, 4]  xyxy pixel (orig image)
        confidences = outputs[2][0]                # [N]

        mask = confidences >= conf_threshold
        class_ids, boxes, confidences = class_ids[mask], boxes[mask], confidences[mask]

        results = []
        for i in range(len(class_ids)):
            x1, y1, x2, y2 = boxes[i]
            # 정규화 좌표로 변환 (orig image 기준)
            results.append(Detection(
                x=float(x1 / orig_w),
                y=float(y1 / orig_h),
                w=float((x2 - x1) / orig_w),
                h=float((y2 - y1) / orig_h),
                class_id=int(class_ids[i]),
                confidence=float(confidences[i]),
            ))
        return results
