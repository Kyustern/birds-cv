"""birds_cv: modular YOLO11n NCNN video inference pipeline.

The package is split into focused modules:

- config:    default paths, colors and thresholds
- tracker:   TemporalSmoother for cross-frame detection consistency
- enhance:   CLAHE + unsharp-masking image enhancement
- drawing:   bounding-box and HUD overlay rendering
- camera:    Raspberry Pi camera source via the libcamera (rpicam-vid) stack
- pipeline:  the run_inference() orchestration loop

The command-line entry point lives in src/video_inference_ncnn.py.
"""

from .config import (
    BBOX_COLORS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_VIDEO_PATH,
)
from .tracker import TemporalSmoother
from .enhance import enhance_frame
from .drawing import draw_detection, draw_info_overlay
from .camera import start_rpicam_source, stop_rpicam_source
from .pipeline import run_inference

__all__ = [
    'BBOX_COLORS',
    'DEFAULT_CONFIDENCE_THRESHOLD',
    'DEFAULT_MODEL_PATH',
    'DEFAULT_OUTPUT_PATH',
    'DEFAULT_VIDEO_PATH',
    'TemporalSmoother',
    'draw_detection',
    'draw_info_overlay',
    'enhance_frame',
    'run_inference',
    'start_rpicam_source',
    'stop_rpicam_source',
]
