#!/usr/bin/env python3

"""
YOLO video inference script using NCNN model with Ultralytics

This script loads a YOLO11n model in NCNN format and runs inference on a video file.
Usage:
    python video_inference_ncnn.py [--headless] [--model MODEL_PATH] [--video VIDEO_PATH] [--output OUTPUT_PATH]

Examples:
    python video_inference_ncnn.py                          # Interactive mode with GUI
    python video_inference_ncnn.py --headless                # Headless mode (no GUI)
    python video_inference_ncnn.py --video sample_vids/8170-207209141_small.mp4
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
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

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run YOLO11n NCNN model inference on video files'
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
    
    return parser.parse_args()


def main():
    """Main function to run video inference."""
    
    args = parse_args()
    
    # Use command line arguments or defaults
    MODEL_PATH = args.model
    VIDEO_PATH = args.video
    OUTPUT_PATH = args.output
    HEADLESS = args.headless
    
    print("=" * 60)
    print("YOLO11n NCNN Video Inference")
    if HEADLESS:
        print("(Headless Mode)")
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
        
        # Convert frame to RGB (Ultralytics expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference with the NCNN model
        try:
            results = model.predict(frame_rgb, verbose=False, conf=CONFIDENCE_THRESHOLD)
            
            # Process results
            if results and len(results) > 0:
                detections = results[0].boxes
                
                # Draw detections on frame
                for i in range(len(detections)):
                    # Get bounding box coordinates
                    xyxy_tensor = detections[i].xyxy.cpu()
                    xyxy = xyxy_tensor.numpy().squeeze()
                    xmin, ymin, xmax, ymax = xyxy.astype(int)
                    
                    # Ensure coordinates are within frame bounds
                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(frame_width, xmax), min(frame_height, ymax)
                    
                    # Get class ID and name
                    classidx = int(detections[i].cls.item())
                    classname = labels[classidx]
                    
                    # Get confidence
                    conf = detections[i].conf.item()
                    
                    # Draw bounding box if confidence is high enough
                    if conf > CONFIDENCE_THRESHOLD:
                        color = BBOX_COLORS[classidx % len(BBOX_COLORS)]
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        
                        label = f'{classname}: {int(conf*100)}%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_ymin = max(ymin, labelSize[1] + 10)
                        
                        # Draw label background
                        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), 
                                    (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                        # Draw label text
                        cv2.putText(frame, label, (xmin, label_ymin-7), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        total_detections += 1
                        
                        # Track class counts
                        if classname not in class_counts:
                            class_counts[classname] = 0
                        class_counts[classname] += 1
                        
        except Exception as e:
            print(f"Error during inference on frame {frame_count}: {e}")
        
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
        
        # Write frame to output video
        if out.isOpened():
            out.write(frame)
        
        # Headless: Print progress periodically
        if HEADLESS and frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames | "
                  f"Detections: {total_detections} | "
                  f"Current FPS: {avg_fps:.2f}")
        
        # Interactive: Display frame and check for user input
        if not HEADLESS:
            cv2.imshow('YOLO11n NCNN Video Inference', frame)
            
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
    
    print(f"\nOutput video saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()