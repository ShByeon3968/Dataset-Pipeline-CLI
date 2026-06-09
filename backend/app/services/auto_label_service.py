"""
AI Auto-labeling service using YOLO-World (open-vocabulary object detection).

YOLO-World uses text prompts to detect objects in images and returns bounding boxes.
It is part of the ultralytics package — no additional dependencies required.

Model: yolov8x-worldv2.pt (auto-downloaded by ultralytics on first use)
Output: bounding boxes only (no segmentation masks)

WHY YOLO-WORLD OVER SAM3:
  - No additional dependencies (SAM3 required OpenAI CLIP via git)
  - No ultralytics internal patching needed
  - Direct bbox output — no mask-to-bbox conversion
  - Faster inference
  - More stable API
"""
import logging
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

logger = logging.getLogger(__name__)

MODEL_DIR = Path("./data/models")
MODEL_NAME = "yolov8x-worldv2.pt"

# Global model cache (loaded once per process)
_model = None


def _get_model_path() -> str:
    """Return local model path if cached, else model name for auto-download."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local = MODEL_DIR / MODEL_NAME
    if local.exists():
        return str(local)
    return MODEL_NAME


def _get_model():
    """Load YOLOWorld once and cache in process memory."""
    global _model
    if _model is not None:
        return _model

    try:
        from ultralytics import YOLOWorld
    except ImportError as e:
        raise RuntimeError(
            "ultralytics is not installed or YOLOWorld is unavailable. "
            "Please rebuild the Docker image."
        ) from e

    model_path = _get_model_path()
    _model = YOLOWorld(model_path)
    logger.info("YOLOWorld model loaded from: %s", model_path)

    _cache_downloaded_model()
    return _model


def _cache_downloaded_model():
    """If ultralytics downloaded the model to cwd, move it to data/models/."""
    import shutil
    cwd_model = Path(MODEL_NAME)
    target = MODEL_DIR / MODEL_NAME
    if cwd_model.exists() and not target.exists():
        try:
            shutil.move(str(cwd_model), str(target))
            logger.info("Moved %s to %s", MODEL_NAME, target)
        except Exception as e:
            logger.warning("Could not move model file: %s", e)


def predict_image(
    image_path: str,
    text_prompts: list[str],
    confidence_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Run YOLO-World text-prompted detection on an image file.

    Args:
        image_path: Absolute path to the image file.
        text_prompts: List of class names, e.g. ["person", "car"].
        confidence_threshold: Minimum confidence score to keep a detection.

    Returns:
        List of dicts with keys:
            class_name (str)      — matched text prompt
            bbox (dict)           — {x, y, w, h} normalized, top-left origin
            confidence (float)    — detection confidence
            segmentation (None)   — always None; YOLO-World is bbox-only

    BBOX FORMAT:
        x, y = top-left corner (normalized 0~1)
        w, h = width/height (normalized 0~1)
        This matches what exporter.py expects for COCO/YOLO/VOC export.
    """
    if not text_prompts:
        raise ValueError("text_prompts must not be empty")

    model = _get_model()

    # set_classes updates the text encoder embeddings for the given prompts.
    # Must be called before every predict() when prompts change.
    model.set_classes(text_prompts)

    results = model.predict(source=image_path, conf=confidence_threshold, verbose=False)

    if not results:
        return []

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    img = PILImage.open(image_path)
    img_w, img_h = img.size

    detections = []
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue

        cls_idx = int(box.cls[0])
        class_name = text_prompts[cls_idx] if cls_idx < len(text_prompts) else text_prompts[0]

        # xyxy: absolute pixel coordinates [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        if bw <= 0 or bh <= 0:
            continue

        detections.append({
            "class_name": class_name,
            "bbox": {
                "x": x1 / img_w,   # top-left x normalized
                "y": y1 / img_h,   # top-left y normalized
                "w": bw,
                "h": bh,
            },
            "confidence": conf,
            "segmentation": None,   # YOLO-World outputs bboxes only
        })

    return detections
