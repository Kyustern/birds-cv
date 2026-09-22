"""Inference orchestration: load model, run the per-frame loop, write output."""

import os
import sys
import threading
import time
import atexit
import subprocess

import cv2
import numpy as np
from ultralytics import YOLO

from .config import (
    BBOX_COLORS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_HEADLESS_OUTPUT_DIR,
    DEFAULT_WS_URL,
)
from .tracker import TemporalSmoother
from .enhance import enhance_frame
from .drawing import draw_detection, draw_info_overlay
from .camera import start_rpicam_source, stop_rpicam_source
from .websocket_client import WebSocketEmitter
from .stream_server import StreamServer, DEFAULT_STREAM_HOST, DEFAULT_STREAM_PORT
from .frame_reader import LatestFrameReader

PUBLISHER_INTERVAL = 5000

def resolve_output_path(args):
    """
    Resolve the output video path.

    When --output is left at its default sentinel, generate a timestamped
    name OUT_<date>_<HHMM>.mp4 inside the headless_output directory so each run
    produces a distinct, self-describing file. Otherwise use the provided path
    and ensure its directory exists.
    """
    if args.output == DEFAULT_OUTPUT_PATH:
        output_dir = DEFAULT_HEADLESS_OUTPUT_DIR
        timestamp = time.strftime("%Y%m%d_%H%M")
        output_path = os.path.join(output_dir, f'OUT_{timestamp}.mp4')
    else:
        output_path = args.output
        output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    return output_path


def extract_detections(results, labels, frame_width, frame_height, conf_threshold):
    """
    Convert Ultralytics results for a single frame into a list of detection dicts.

    Each dict has 'xyxy' (clamped to frame bounds), 'conf', 'cls' and 'classname'.
    Only detections above conf_threshold are included.
    """
    detections_out = []
    if not results or len(results) == 0:
        return detections_out

    detections = results[0].boxes
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze()

        # Ensure coordinates are within frame bounds
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(frame_width, xmax), min(frame_height, ymax)
        xyxy = np.array([xmin, ymin, xmax, ymax])

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > conf_threshold:
            detections_out.append({
                'xyxy': xyxy,
                'conf': conf,
                'cls': classidx,
                'classname': classname,
            })
    return detections_out

# def publisher_loop() :
    # global heartbeat_active
    
    # while heartbeat_active:
        # with heartbeat_lock:
            # Getting the freshest prediction data
            # print("publishing..")
            # predictions = kalman_filter_service.get_latest_predictions()

            # # Send heartbeat to all connected clients
            # for sid in list(connected_clients):
            #     socketio.emit('heartbeat', time.time(), room=sid)
        
        # Sleep for the heartbeat interval
        # time.sleep(PUBLISHER_INTERVAL)
    

# publisher_thread= threading.Thread(
#     target=publisher_loop,
#     daemon=True,
#     name="websocket-heartbeat",
# )

def run_inference(args):
    """Load the model and run the inference loop over the configured video source."""

    MODEL_PATH = args.model
    VIDEO_PATH = args.video
    HEADLESS = args.headless
    ENABLE_ENHANCE = args.enhance
    ENABLE_TEMPORAL = args.temporal
    CONFIDENCE_THRESHOLD = args.conf
    OUTPUT_PATH = resolve_output_path(args)

    print("=" * 60)
    print("YOLO11n NCNN Video Inference with Improvements")
    if HEADLESS:
        print("(Headless Mode)")
    improvements = []
    if ENABLE_ENHANCE:
        improvements.append("Image Enhancement")
    if ENABLE_TEMPORAL:
        improvements.append("Temporal Smoothing")
    if improvements:
        print(f"Enabled: {', '.join(improvements)}")
    print("=" * 60)

    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f'ERROR: Model path is invalid or model was not found: {MODEL_PATH}')
        sys.exit(1)

    print(f"Loading YOLO model from: {MODEL_PATH}")

    # Load the NCNN model with Ultralytics
    try:
        model = YOLO(MODEL_PATH, task='detect')
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        labels = model.names
        print(f"Number of classes: {len(labels)}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Initialize temporal smoother if enabled
    temporal_smoother = TemporalSmoother(max_frames=5, iou_threshold=0.5) if ENABLE_TEMPORAL else None

    # Connect the WebSocket emitter (if a server URL was provided) before the
    # inference loop starts, so per-frame detections can be streamed out.
    print("args.ws_url", args.ws_url)

    url = args.ws_url or DEFAULT_WS_URL
    print("url", url)
    ws_emitter = WebSocketEmitter(
        url=url
    )
    if ws_emitter is not None:
        ws_emitter.connect()

    # Open the video source: a file, or the Raspberry Pi camera via libcamera
    rpicam_proc = None
    if args.rpicam:
        VIDEO_PATH, rpicam_proc = start_rpicam_source(
            width=args.width, height=args.height, framerate=args.framerate
        )
        if VIDEO_PATH is None:
            sys.exit(1)

    print(f"Opening video source: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {VIDEO_PATH}")
        stop_rpicam_source(rpicam_proc)
        sys.exit(1)

    # Get video properties (these are 0/unknown for a live camera pipe)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    live_stream = (total_frames <= 0)
    if fps <= 0:
        fps = float(args.framerate) if args.rpicam else 30.0

    frames_desc = 'live' if live_stream else str(total_frames)
    print(f"Video properties: {frame_width}x{frame_height}, {fps:.2f} FPS, {frames_desc} frames")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    # Set up video writer for output (created now if dims are known; deferred for live streams)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    if frame_width > 0 and frame_height > 0:
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            print("WARNING: Could not open output video writer")
            out = None

    # Ensure the capture, writer, and rpicam-vid subprocess are cleaned up even when
    # the live stream is stopped with Ctrl-C or killed by a signal (otherwise the
    # output .mp4 is left without a moov atom and is unplayable).
    # frame_reader is populated later for live streams; declared here so the
    # cleanup closure always sees it bound.
    frame_reader = None

    def _cleanup_source():
        try:
            cap.release()
        except Exception:
            pass
        if out is not None:
            try:
                out.release()
            except Exception:
                pass
        if frame_reader is not None:
            frame_reader.stop()
        stop_rpicam_source(rpicam_proc)
        if ws_emitter is not None:
            ws_emitter.close()

    atexit.register(_cleanup_source)
    try:
        import signal

        def _term_handler(signum, _frame):
            _cleanup_source()
            sys.exit(0)
        signal.signal(signal.SIGINT, _term_handler)
        signal.signal(signal.SIGTERM, _term_handler)
    except Exception:
        pass

    print("Starting video inference...")
    if not HEADLESS:
        print("Press 'q' to quit, 's' to pause, 'p' to save current frame")

    # Optionally serve the annotated frames as a live MJPEG stream so the
    # running pipeline can be viewed in a browser, mirroring start_stream.py.
    stream_server = None
    if getattr(args, 'stream', False):
        stream_host = getattr(args, 'stream_host', DEFAULT_STREAM_HOST) or DEFAULT_STREAM_HOST
        stream_port = getattr(args, 'stream_port', DEFAULT_STREAM_PORT)
        stream_server = StreamServer(host=stream_host, port=stream_port)
        stream_server.start()
    if stream_server is not None:
        print("Live annotated frames will be available at /stream and /snapshot")
    print("-" * 60)

    frame_count = 0
    processing_times = []
    total_detections = 0
    class_counts = {}

    inference_start = time.perf_counter()

    # For a live camera source, read frames on a background thread that keeps
    # only the newest frame. The inference loop then always runs on the freshest
    # available frame and never falls behind processing a backlog of stale
    # "lost" frames. Recorded files keep the sequential cap.read() so every
    # frame is processed in order.
    last_seq = 0
    if live_stream:
        frame_reader = LatestFrameReader(cap)
        frame_reader.start()

    # _ _ _ _ _ _ _ _ _ _ Main inference loop _ _ _ _ _ _ _ _ _ _
    while cap.isOpened():
        if frame_reader is not None:
            seq, frame = frame_reader.next_fresh(last_seq)
            if frame is None:
                break
            last_seq = seq
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame = cv2.flip(frame, 0)

        # For a live camera pipe the CAP properties are unknown, so derive frame
        # dimensions from the frame itself (before inference uses them) and lazily
        # create the output writer.
        if frame_width <= 0 or frame_height <= 0:
            frame_height, frame_width = frame.shape[:2]
            if out is None:
                out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
                if not out.isOpened():
                    print("WARNING: Could not open output video writer")
                    out = None

        frame_count += 1
        frame_start = time.perf_counter()

        # Apply image enhancement if enabled
        if ENABLE_ENHANCE:
            frame = enhance_frame(frame)

        # Convert frame to RGB (Ultralytics expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference with the NCNN model
        raw_detections = []
        try:
            results = model.predict(frame_rgb, verbose=False, conf=CONFIDENCE_THRESHOLD)
            raw_detections = extract_detections(
                results, labels, frame_width, frame_height, CONFIDENCE_THRESHOLD
            )
        except Exception as e:
            print(f"Error during inference on frame {frame_count}: {e}")

        # Apply temporal smoothing if enabled
        if ENABLE_TEMPORAL and temporal_smoother:
            smoothed_detections = temporal_smoother.update(raw_detections, frame_count)
        else:
            smoothed_detections = raw_detections

        # Stream detections to the WebSocket server (bounding boxes + center points)
        if ws_emitter is not None:
            ws_emitter.emit_detections(smoothed_detections, frame_count)

        # Draw smoothed detections on frame
        for det in smoothed_detections:
            xyxy = det['xyxy']
            classname = det['classname']
            conf = det['conf']
            classidx = det['cls']

            color = BBOX_COLORS[classidx % len(BBOX_COLORS)]
            draw_detection(frame, xyxy, classname, conf, color)

            total_detections += 1
            if classname not in class_counts:
                class_counts[classname] = 0
            class_counts[classname] += 1

        # Calculate processing statistics
        processing_time = time.perf_counter() - frame_start
        processing_times.append(processing_time)
        avg_fps = len(processing_times) / sum(processing_times) if processing_times else 0

        # Draw HUD overlay on frame
        active_tracks = len(temporal_smoother.get_active_tracks()) if temporal_smoother else 0
        draw_info_overlay(frame, frame_count, total_frames, live_stream, avg_fps,
                          total_detections, ENABLE_ENHANCE, ENABLE_TEMPORAL, active_tracks)

        # Write frame to output video
        if out is not None and out.isOpened():
            out.write(frame)

        # Push the annotated frame to the live MJPEG stream (if enabled)
        if stream_server is not None:
            stream_server.update_frame(frame)

        # Headless: Print progress periodically
        if HEADLESS and frame_count % 10 == 0:
            progress = f'{frame_count} frames' if live_stream else f'{frame_count}/{total_frames} frames'
            print(f"Processed {progress} | "
                  f"Detections: {total_detections} | "
                  f"Tracks: {active_tracks} | "
                  f"Current FPS: {avg_fps:.2f}")

        # end_time = time.perf_counter()
        
        # Interactive: Display frame and check for user input
        if not HEADLESS:
            cv2.imshow('YOLO11n NCNN Video Inference with Improvements', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User requested to quit.")
                break
            elif key == ord('s'):
                print("Paused. Press any key to continue...")
                cv2.waitKey(0)  # Wait for any key to continue
            elif key == ord('p'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                frame_filename = f'frame_{frame_count:04d}_{timestamp}.jpg'
                cv2.imwrite(frame_filename, frame)
                print(f"Saved frame to: {frame_filename}")

    # Clean up
    cap.release()
    if out is not None:
        out.release()
    if frame_reader is not None:
        frame_reader.stop()
    stop_rpicam_source(rpicam_proc)
    if ws_emitter is not None:
        ws_emitter.close()

    if not HEADLESS:
        cv2.destroyAllWindows()

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    processed_desc = str(frame_count) if live_stream else f'{frame_count}/{total_frames}'
    print(f"Total frames processed: {processed_desc}")
    print(f"Total detections: {total_detections}")

    if class_counts:
        print("\nDetection counts by class:")
        for classname, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {classname}: {count}")

    if processing_times:
        avg_fps = len(processing_times) / sum(processing_times)
        avg_time = np.mean(processing_times) * 1000
        min_time = np.min(processing_times) * 1000
        max_time = np.max(processing_times) * 1000
        print(f"\nPerformance:")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Average processing time per frame: {avg_time:.2f} ms")
        print(f"  Min processing time: {min_time:.2f} ms")
        print(f"  Max processing time: {max_time:.2f} ms")

    if ENABLE_TEMPORAL and temporal_smoother:
        active_tracks = len(temporal_smoother.get_active_tracks())
        print(f"\nTemporal Smoothing:")
        print(f"  Active tracks at end: {active_tracks}")

    print(f"\nOutput video saved to: {OUTPUT_PATH}")
    print("=" * 60)
