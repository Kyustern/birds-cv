"""birds_cv: modular YOLO11n NCNN video inference pipeline.

The package is split into focused modules:

- config:    default paths, colors and thresholds
- tracker:   TemporalSmoother for cross-frame detection consistency
- enhance:   CLAHE + unsharp-masking image enhancement
- drawing:   bounding-box and HUD overlay rendering
- camera:    Raspberry Pi camera source via the libcamera (rpicam-vid) stack
- pipeline:  the run_inference() orchestration loop
- websocket_client: stream per-frame detections to a WebSocket server
- stream_server: serve annotated frames as an MJPEG stream over HTTP

The command-line entry point lives in src/video_inference_ncnn.py.
"""

from .config import (
    BBOX_COLORS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_VIDEO_PATH,
    DEFAULT_WS_URL,
)
from .tracker import TemporalSmoother
from .enhance import enhance_frame
from .drawing import draw_detection, draw_info_overlay
from .camera import start_rpicam_source, stop_rpicam_source
from .websocket_client import WebSocketEmitter
from .stream_server import (
    StreamServer,
    DEFAULT_STREAM_HOST,
    DEFAULT_STREAM_PORT,
)
from .frame_reader import LatestFrameReader
from .pipeline import run_inference

__all__ = [
    'BBOX_COLORS',
    'DEFAULT_STREAM_HOST',
    'DEFAULT_STREAM_PORT',
    'DEFAULT_CONFIDENCE_THRESHOLD',
    'DEFAULT_MODEL_PATH',
    'DEFAULT_OUTPUT_PATH',
    'DEFAULT_VIDEO_PATH',
    'DEFAULT_WS_URL',
    'StreamServer',
    'TemporalSmoother',
    'LatestFrameReader',
    'WebSocketEmitter',
    'draw_detection',
    'draw_info_overlay',
    'enhance_frame',
    'run_inference',
    'start_rpicam_source',
    'stop_rpicam_source',
]
