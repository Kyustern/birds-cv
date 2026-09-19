#!/usr/bin/env python3

"""
YOLO video inference entry point using NCNN model with Ultralytics.

The implementation lives in the birds_cv package; this script just parses the
command-line arguments and hands off to birds_cv.pipeline.run_inference().

Usage:
    python video_inference_ncnn.py [--headless] [--model MODEL_PATH] [--video VIDEO_PATH]
                                    [--output OUTPUT_PATH] [--enhance] [--temporal]
                                    [--conf CONF] [--rpicam] [--width W] [--height H]
                                    [--framerate FPS]

Examples:
    python video_inference_ncnn.py                          # Interactive mode with GUI
    python video_inference_ncnn.py --headless                # Headless mode (no GUI)
    python video_inference_ncnn.py --video sample_vids/8170-207209141_small.mp4
    python video_inference_ncnn.py --headless --enhance --temporal  # Both improvements enabled
    python video_inference_ncnn.py --headless --rpicam       # Raspberry Pi camera via libcamera
"""

import argparse

from birds_cv import (
    run_inference,
    DEFAULT_MODEL_PATH,
    DEFAULT_VIDEO_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run YOLO11n NCNN model inference on video files with improvements'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode without displaying the video (useful for servers)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f'Path to the model folder (default: {DEFAULT_MODEL_PATH})'
    )

    parser.add_argument(
        '--video',
        type=str,
        default=DEFAULT_VIDEO_PATH,
        help=f'Path to the input video file (default: {DEFAULT_VIDEO_PATH})'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help='Output video path. If left as default, a timestamped OUT_<date>_<HHMM>.mp4 '
             'is written to headless_output/.'
    )

    parser.add_argument(
        '--enhance',
        action='store_true',
        help='Enable image enhancement (CLAHE + sharpening) for better detection'
    )

    parser.add_argument(
        '--temporal',
        action='store_true',
        help='Enable temporal smoothing to track detections across frames'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f'Confidence threshold for detections (default: {DEFAULT_CONFIDENCE_THRESHOLD})'
    )

    parser.add_argument(
        '--rpicam',
        action='store_true',
        help='Use the Raspberry Pi camera via libcamera (rpicam-vid) instead of a video file'
    )

    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Camera output width in pixels (default: 640, used with --rpicam)'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Camera output height in pixels (default: 480, used with --rpicam)'
    )

    parser.add_argument(
        '--framerate',
        type=int,
        default=30,
        help='Camera framerate in fps (default: 30, used with --rpicam)'
    )

    return parser.parse_args()


def main():
    """Parse arguments and run the inference pipeline."""
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
