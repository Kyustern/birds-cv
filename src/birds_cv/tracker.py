"""Cross-frame detection tracking for temporal consistency."""

from collections import defaultdict


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
