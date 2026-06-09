"""
ONNX 모델 추론 서비스

세션 캐시, 텐서 검증, 아키텍처별 어댑터 호출을 담당합니다.
"""
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_sessions: dict[int, Any] = {}   # model_id -> ort.InferenceSession


def _abs(file_path: str, models_dir: str) -> str:
    p = Path(file_path)
    if p.is_absolute():
        return str(p)
    return str(Path(models_dir) / file_path)


def get_session(model_id: int, file_path: str, models_dir: str):
    if model_id not in _sessions:
        import onnxruntime as ort
        abs_path = _abs(file_path, models_dir)
        sess = ort.InferenceSession(
            abs_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _sessions[model_id] = sess
        logger.info("ONNX 세션 로드: model_id=%d  path=%s", model_id, abs_path)
    return _sessions[model_id]


def evict_session(model_id: int) -> None:
    _sessions.pop(model_id, None)


def validate_model(abs_path: str) -> dict:
    """업로드 직후 입출력 텐서 정보 반환."""
    import onnxruntime as ort
    sess = ort.InferenceSession(abs_path, providers=["CPUExecutionProvider"])
    return {
        "inputs": [
            {"name": i.name, "shape": list(i.shape), "dtype": i.type}
            for i in sess.get_inputs()
        ],
        "outputs": [
            {"name": o.name, "shape": list(o.shape), "dtype": o.type}
            for o in sess.get_outputs()
        ],
    }


def run_inference(
    model_id: int,
    file_path: str,
    models_dir: str,
    architecture: str,
    class_labels: list[str],
    image_abs_path: str,
    input_w: int,
    input_h: int,
    conf_threshold: float,
    iou_threshold: float,
) -> list[dict]:
    import cv2
    from app.services.detectors.factory import get_detector

    bgr = cv2.imread(image_abs_path)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {image_abs_path}")

    orig_h, orig_w = bgr.shape[:2]
    detector = get_detector(architecture)
    tensor = detector.preprocess(bgr, input_w, input_h)

    sess = get_session(model_id, file_path, models_dir)

    # 모든 필수 입력을 수집하여 feed dict 구성
    # DETR 계열 모델은 images 외에 orig_target_sizes [1,2] 를 추가로 요구함
    input_names = {inp.name for inp in sess.get_inputs()}
    feed: dict[str, np.ndarray] = {sess.get_inputs()[0].name: tensor}

    if "orig_target_sizes" in input_names:
        # shape [1, 2] — (height, width) int64
        feed["orig_target_sizes"] = np.array([[orig_h, orig_w]], dtype=np.int64)
        logger.debug("orig_target_sizes 주입: [%d, %d]", orig_h, orig_w)

    outputs = sess.run(None, feed)

    detections = detector.postprocess(
        outputs, orig_w, orig_h, input_w, input_h, conf_threshold, iou_threshold
    )

    results = []
    for d in detections:
        cname = (
            class_labels[d.class_id]
            if d.class_id < len(class_labels)
            else f"class_{d.class_id}"
        )
        results.append({
            "class_name": cname,
            "bbox": {"x": d.x, "y": d.y, "w": d.w, "h": d.h},
            "confidence": d.confidence,
            "segmentation": None,
        })
    return results
