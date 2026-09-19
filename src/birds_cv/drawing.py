"""Bounding-box and HUD overlay drawing on video frames."""

import cv2


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


def draw_info_overlay(frame, frame_count, total_frames, live_stream, avg_fps,
                      total_detections, enable_enhance, enable_temporal, active_tracks):
    """Draw the on-frame HUD: frame counter, FPS, detection count and feature status."""
    frame_info = f'Frame: {frame_count}' if live_stream else f'Frame: {frame_count}/{total_frames}'
    cv2.putText(frame, frame_info, (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'FPS: {avg_fps:.2f}', (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Detections: {total_detections}', (20, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Add enhancement/temporal indicators
    status_y = 120
    if enable_enhance:
        cv2.putText(frame, 'Enhancement: ON', (20, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        status_y += 30
    if enable_temporal:
        cv2.putText(frame, f'Temporal: ON | Tracks: {active_tracks}', (20, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
