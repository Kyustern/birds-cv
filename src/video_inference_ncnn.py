#!/usr/bin/env python3

"""
YOLO video inference script using NCNN model with Ultralytics

This script loads a YOLO11n model in NCNN format and runs inference on a video file.
Usage:
    python video_inference_ncnn.py [--headless] [--model MODEL_PATH] [--video VIDEO_PATH] [--output OUTPUT_PATH]
    [--enhance] [--temporal] [--conf CONFIDENCE]

Examples:
    python video_inference_ncnn.py                          # Interactive mode with GUI
    python video_inference_ncnn.py --headless                # Headless mode (no GUI)
    python video_inference_ncnn.py --video sample_vids/8170-207209141_small.mp4
    python video_inference_ncnn.py --headless --enhance --temporal  # Both improvements enabled
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Default paths - easily configurable
DEFAULT_MODEL_PATH = '../yolo11n_ncnn_model'  # Path to NCNN model folder
DEFAULT_VIDEO_PATH = '../sample_vids/8170-207209141_small.mp4'  # Path to video file
DEFAULT_OUTPUT_PATH = '../headless_output/output.mp4'  # Output video path with bounding boxes

# Bounding box colors (Tableau 10 color scheme)
BBOX_COLORS = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
    (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
]

# Default detection parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


class TemporalSmoother:
    """Track detections across frames to improve consistency."""
    
    def __init__(self, max_frames=5, iou_threshold=0.5):
        """
        Initialize the temporal smoother.
        
        Args:
            max_frames: Maximum number of frames to track a detection without updates
            iou_threshold: IOU threshold to consider detections as the same object
        """
        self.tracks = defaultdict(list)  # class_id -> list of track dicts
        self.max_frames = max_frames
        self.iou_threshold = iou_threshold
        self.current_frame = 0
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IOU) between two boxes."""
        x1, y1, x2, y2 = box1
        fx1, fy1, fx2, fy2 = box2
        
        inter_x1 = max(x1, fx1)
        inter_y1 = max(y1, fy1)
        inter_x2 = min(x2, fx2)
        inter_y2 = min(y2, fy2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (fx2 - fx1) * (fy2 - fy1)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections, frame_count):
        """
        Update tracker with current frame detections.
        
        Args:
            detections: List of detection dicts with 'xyxy', 'conf', 'cls', 'classname'
            frame_count: Current frame number
        
        Returns:
            List of smoothed detections
        """
        self.current_frame = frame_count
        
        # For each class, match current detections with existing tracks
        for classidx, tracks in list(self.tracks.items()):
            current_class_dets = [d for d in detections if d['cls'] == classidx]
            
            for track in tracks:
                best_match_idx = None
                best_iou = 0
                
                for i, det in enumerate(current_class_dets):
                    iou = self.calculate_iou(track['xyxy'], det['xyxy'])
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_match_idx = i
                
                if best_match_idx is not None:
                    # Update track with matched detection
                    matched_det = current_class_dets[best_match_idx]
                    track['xyxy'] = matched_det['xyxy']
                    track['conf'] = matched_det['conf']
                    track['frame'] = frame_count
                    track['classname'] = matched_det['classname']
                    # Mark as matched
                    current_class_dets[best_match_idx]['_matched'] = True
            
            # Add unmatched detections as new tracks
            for det in current_class_dets:
                if not det.get('_matched', False):
                    self.tracks[classidx].append({
                        'xyxy': det['xyxy'],
                        'conf': det['conf'],
                        'cls': det['cls'],
                        'classname': det['classname'],
                        'frame': frame_count
                    })
            
            # Remove old tracks (no updates for max_frames)
            self.tracks[classidx] = [
                t for t in self.tracks[classidx] 
                if frame_count - t['frame'] <= self.max_frames
            ]
        
        # Return all current active tracks as smoothed detections
        smoothed_detections = []
        for classidx, tracks in self.tracks.items():
            for track in tracks:
                smoothed_detections.append({
                    'xyxy': track['xyxy'],
                    'conf': track['conf'],
                    'cls': track['cls'],
                    'classname': track['classname']
                })
        
        return smoothed_detections
    
    def get_active_tracks(self):
        """Get all currently active tracks."""
        all_tracks = []
        for classidx, tracks in self.tracks.items():
            all_tracks.extend(tracks)
        return all_tracks


def enhance_frame(frame, clip_limit=2.0, tile_size=(8, 8)):
    """
    Apply image enhancement to improve detection quality.
    
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast
    enhancement and unsharp masking for sharpening.
    
    Args:
        frame: Input BGR frame
        clip_limit: CLAHE clip limit (higher = more contrast)
        tile_size: CLAHE grid size
    
    Returns:
        Enhanced BGR frame
    """
    # Convert to LAB color space for better contrast manipulation
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    cl = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Apply sharpening using unsharp masking
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    
    return sharpened


def draw_detection(frame, xyxy, classname, conf, color):
    """Draw a single detection on the frame."""
    xmin, ymin, xmax, ymax = xyxy.astype(int)
    
    # Draw bounding box
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    
    # Draw label
    label = f'{classname}: {int(conf*100)}%'
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_ymin = max(ymin, labelSize[1] + 10)
    
    # Draw label background
    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), 
                (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
    # Draw label text
    cv2.putText(frame, label, (xmin, label_ymin-7), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


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
        help=f'Path to the output video file (default: {DEFAULT_OUTPUT_PATH})'
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
    
    return parser.parse_args()


def main():
    """Main function to run video inference."""
    
    args = parse_args()
    
    # Use command line arguments or defaults
    MODEL_PATH = args.model
    VIDEO_PATH = args.video
    OUTPUT_PATH = args.output
    HEADLESS = args.headless
    ENABLE_ENHANCE = args.enhance
    ENABLE_TEMPORAL = args.temporal
    CONFIDENCE_THRESHOLD = args.conf
    
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
    
    # Check if video path exists
    if not os.path.exists(VIDEO_PATH):
        print(f'ERROR: Video path is invalid or video was not found: {VIDEO_PATH}')
        sys.exit(1)
    
    print(f"Loading YOLO model from: {MODEL_PATH}")
    
    # Load the NCNN model with Ultralytics
    try:
        model = YOLO(MODEL_PATH, task='detect')
        print("✓ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        labels = model.names
        print(f"Number of classes: {len(labels)}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)
    
    # Initialize temporal smoother if enabled
    temporal_smoother = TemporalSmoother(max_frames=5, iou_threshold=0.5) if ENABLE_TEMPORAL else None
    
    # Open the video file
    print(f"Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {VIDEO_PATH}")
        sys.exit(1)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} frames")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    # Set up video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("WARNING: Could not open output video writer")
    
    print("Starting video inference...")
    if not HEADLESS:
        print("Press 'q' to quit, 's' to pause, 'p' to save current frame")
    print("-" * 60)
    
    frame_count = 0
    processing_times = []
    total_detections = 0
    class_counts = {}
    
    # Main inference loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.perf_counter()
        
        # Apply image enhancement if enabled
        if ENABLE_ENHANCE:
            frame = enhance_frame(frame)
        
        # Convert frame to RGB (Ultralytics expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference with the NCNN model
        raw_detections = []
        try:
            results = model.predict(frame_rgb, verbose=False, conf=CONFIDENCE_THRESHOLD)
            
            # Process results
            if results and len(results) > 0:
                detections = results[0].boxes
                
                # Convert to list of dicts for easier processing
                for i in range(len(detections)):
                    xyxy_tensor = detections[i].xyxy.cpu()
                    xyxy = xyxy_tensor.numpy().squeeze()
                    
                    # Ensure coordinates are within frame bounds
                    xmin, ymin, xmax, ymax = xyxy.astype(int)
                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(frame_width, xmax), min(frame_height, ymax)
                    xyxy = np.array([xmin, ymin, xmax, ymax])
                    
                    classidx = int(detections[i].cls.item())
                    classname = labels[classidx]
                    conf = detections[i].conf.item()
                    
                    if conf > CONFIDENCE_THRESHOLD:
                        raw_detections.append({
                            'xyxy': xyxy,
                            'conf': conf,
                            'cls': classidx,
                            'classname': classname
                        })
        except Exception as e:
            print(f"Error during inference on frame {frame_count}: {e}")
        
        # Apply temporal smoothing if enabled
        if ENABLE_TEMPORAL and temporal_smoother:
            smoothed_detections = temporal_smoother.update(raw_detections, frame_count)
        else:
            smoothed_detections = raw_detections
        
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
        processing_time = time.perf_counter() - start_time
        processing_times.append(processing_time)
        avg_fps = len(processing_times) / sum(processing_times) if processing_times else 0
        
        # Draw info on frame
        cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {avg_fps:.2f}', (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Detections: {total_detections}', (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add enhancement/temporal indicators
        status_y = 120
        if ENABLE_ENHANCE:
            cv2.putText(frame, 'Enhancement: ON', (20, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            status_y += 30
        if ENABLE_TEMPORAL:
            active_tracks = len(temporal_smoother.get_active_tracks()) if temporal_smoother else 0
            cv2.putText(frame, f'Temporal: ON | Tracks: {active_tracks}', (20, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame to output video
        if out.isOpened():
            out.write(frame)
        
        # Headless: Print progress periodically
        if HEADLESS and frame_count % 100 == 0:
            active_tracks = len(temporal_smoother.get_active_tracks()) if temporal_smoother else 0
            print(f"Processed {frame_count}/{total_frames} frames | "
                  f"Detections: {total_detections} | "
                  f"Tracks: {active_tracks} | "
                  f"Current FPS: {avg_fps:.2f}")
        
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
    if out.isOpened():
        out.release()
    
    if not HEADLESS:
        cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}/{total_frames}")
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


if __name__ == "__main__":
    main()