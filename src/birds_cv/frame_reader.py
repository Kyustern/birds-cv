"""Live frame reader that always yields the freshest captured frame.

For a live camera source (e.g. the rpicam-vid MJPEG pipe) OpenCV's
VideoCapture buffers frames internally, and the OS pipe buffers more. When
inference is slower than the capture rate, cap.read() returns stale,
backlogged frames and the pipeline falls behind real time, "catching up" on
lost frames instead of acting on current data.

LatestFrameReader drains the capture on a background thread and keeps only
the most recent frame. The inference loop asks for a frame newer than the one
it last processed; if several arrived in the meantime, only the newest is
returned and the rest are dropped, so each iteration runs on fresh data.
"""

import threading
import time


class LatestFrameReader:
    """Read frames on a background thread, keeping only the newest.

    The consumer calls next_fresh(last_seq) to obtain a frame newer than the
    one it last processed. If several frames arrived in the meantime, only the
    newest is returned and the rest are discarded; if none has arrived yet,
    the call blocks until one does (or the source reaches end-of-stream).
    """

    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0          # incremented every time a new frame is stored
        self._eof = False     # set when the capture returns no more frames
        self._stopped = False
        self._thread = None

    def start(self):
        """Launch the background reader thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name='frame-reader'
        )
        self._thread.start()

    def _run(self):
        """Continuously read frames, overwriting the stored frame each time."""
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                with self._lock:
                    self._eof = True
                break
            with self._lock:
                self._frame = frame
                self._seq += 1

    def next_fresh(self, last_seq=0):
        """Block until a frame newer than last_seq is available, or end-of-stream.

        Returns (seq, frame) for the newest frame with seq > last_seq, or
        (None, None) when the source has ended and no newer frame remains.
        Intermediate frames produced between calls are discarded.
        """
        while not self._stopped:
            with self._lock:
                if self._seq > last_seq:
                    return self._seq, self._frame
                if self._eof:
                    return None, None
            time.sleep(0.001)
        return None, None

    def stop(self):
        """Signal the reader thread to stop on its next iteration."""
        self._stopped = True
