"""Default configuration constants for the birds-cv inference pipeline."""

# Default paths (relative to src/, where the entry point runs from)
DEFAULT_MODEL_PATH = '../yolo11n_ncnn_model'  # Path to NCNN model folder
DEFAULT_VIDEO_PATH = '../sample_vids/8170-207209141_small.mp4'  # Path to video file
DEFAULT_OUTPUT_PATH = '../headless_output/output.mp4'  # Sentinel; replaced with a timestamped name when left as default
DEFAULT_HEADLESS_OUTPUT_DIR = '../headless_output'  # Directory for timestamped outputs

# Bounding box colors (Tableau 10 color scheme), indexed by class id
BBOX_COLORS = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
    (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184),
]

# Default detection parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# WebSocket server for streaming detections (opt-in via --ws-url)
DEFAULT_WS_URL = 'http://localhost:5000/'

