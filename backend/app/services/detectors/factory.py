from .base import BaseDetector
from .rfdetr import RFDETRDetector
from .deimv2 import DEIMv2Detector

_MAP: dict[str, type[BaseDetector]] = {
    "rfdetr": RFDETRDetector,
    "deimv2": DEIMv2Detector,
}


def get_detector(architecture: str) -> BaseDetector:
    cls = _MAP.get(architecture)
    if not cls:
        raise ValueError(
            f"지원하지 않는 아키텍처: '{architecture}'. "
            f"허용값: {list(_MAP.keys())}"
        )
    return cls()
