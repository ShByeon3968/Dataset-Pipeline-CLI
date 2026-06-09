import numpy as np
from .base import BaseDetector, Detection


class DEIMv2Detector(BaseDetector):
    """
    DEIMv2 ONNX 출력 형식 - orig_target_sizes 전달 여부에 따라 두 가지:

    [형식 A] orig_target_sizes 없이 export
      outputs[0] = 'dets'   [1, 300, 4]   float  xyxy (pixel or normalized)
      outputs[1] = 'labels' [1, 300, nc]  float  class score matrix

    [형식 B] orig_target_sizes 전달 시 (일반적)
      outputs[0] = 'labels' [1, 300]      int64  class_id  <-- ndim==1
      outputs[1] = 'boxes'  [1, 300, 4]   float  xyxy orig pixel (orig_w x orig_h)
      outputs[2] = 'scores' [1, 300]      float  confidence

    outputs[0][0].ndim 으로 형식 판별:
      ndim == 1  ->  형식 B
      ndim == 2  ->  형식 A
    """

    def preprocess(self, bgr: np.ndarray, input_w: int, input_h: int) -> np.ndarray:
        import cv2
        img = cv2.resize(bgr, (input_w, input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[np.newaxis]

    def postprocess(self, outputs, orig_w, orig_h, input_w, input_h, conf_threshold, iou_threshold):
        first = outputs[0][0]   # squeeze batch dim

        if first.ndim == 1:
            # 형식 B: labels(int) + boxes(xyxy pixel) + scores
            return self._postprocess_b(outputs, orig_w, orig_h, conf_threshold)
        else:
            # 형식 A: dets(xyxy) + labels(score matrix)
            return self._postprocess_a(outputs, orig_w, orig_h, input_w, input_h, conf_threshold)

    def _postprocess_a(self, outputs, orig_w, orig_h, input_w, input_h, conf_threshold):
        boxes        = outputs[0][0]   # [300, 4]
        class_scores = outputs[1][0]   # [300, nc]

        class_ids   = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        mask = confidences >= conf_threshold
        boxes       = boxes[mask]
        class_ids   = class_ids[mask]
        confidences = confidences[mask]

        results = []
        for i in range(len(class_ids)):
            x1 = float(boxes[i, 0])
            y1 = float(boxes[i, 1])
            x2 = float(boxes[i, 2])
            y2 = float(boxes[i, 3])
            if x2 > 2.0:   # pixel coords -> normalize by orig size
                x1, x2 = x1 / orig_w, x2 / orig_w
                y1, y2 = y1 / orig_h, y2 / orig_h
            results.append(Detection(
                x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                class_id=int(class_ids[i]),
                confidence=float(confidences[i]),
            ))
        return results

    def _postprocess_b(self, outputs, orig_w, orig_h, conf_threshold):
        class_ids   = outputs[0][0].astype(int)   # [300]
        boxes       = outputs[1][0]                # [300, 4] xyxy orig pixel
        confidences = outputs[2][0].astype(float)  # [300]

        mask = confidences >= conf_threshold
        class_ids   = class_ids[mask]
        boxes       = boxes[mask]
        confidences = confidences[mask]

        results = []
        for i in range(len(class_ids)):
            x1 = float(boxes[i, 0]) / orig_w
            y1 = float(boxes[i, 1]) / orig_h
            x2 = float(boxes[i, 2]) / orig_w
            y2 = float(boxes[i, 3]) / orig_h
            results.append(Detection(
                x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                class_id=int(class_ids[i]),
                confidence=float(confidences[i]),
            ))
        return results
