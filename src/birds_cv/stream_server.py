"""Live MJPEG stream server for the inference pipeline.

Serves the annotated frames produced by run_inference() as a Motion JPEG
stream over HTTP, mirroring start_stream.py but fed by the pipeline rather
than by a standalone camera capture.

    http://<host>:<port>/           -> HTML page embedding the feed
    http://<host>:<port>/stream     -> raw multipart/x-mixed-replace MJPEG stream
    http://<host>:<port>/snapshot   -> single JPEG frame

The pipeline calls update_frame() once per processed frame; a background
thread runs the Flask dev server so streaming does not block inference.
"""

import threading
import time

import cv2
from flask import Flask, Response

DEFAULT_STREAM_HOST = '0.0.0.0'
DEFAULT_STREAM_PORT = 5010


class StreamServer:
    """Hold the latest annotated frame and serve it as an MJPEG stream."""

    def __init__(self, host=DEFAULT_STREAM_HOST, port=DEFAULT_STREAM_PORT):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._frame_bytes = None
        self.app = Flask(__name__)
        self._register_routes()
        self._thread = None

    def update_frame(self, frame):
        """JPEG-encode and store the latest frame for the stream."""
        ok, encoded = cv2.imencode('.jpg', frame)
        if not ok:
            return
        with self._lock:
            self._frame_bytes = encoded.tobytes()

    def _read_frame(self):
        """Return the latest encoded frame bytes, or None if none yet."""
        with self._lock:
            return self._frame_bytes

    def _register_routes(self):
        app = self.app

        @app.route('/')
        def index():
            """Landing page that embeds the live MJPEG feed."""
            return (
                '<html><head><title>birds-cv live stream</title></head>'
                '<body style="margin:0;background:#111">'
                '<img src="/stream" style="width:100%;height:100%;object-fit:contain">'
                '</body></html>'
            )

        @app.route('/stream')
        def stream():
            """Live Motion JPEG stream (multipart/x-mixed-replace)."""
            return Response(self._mjpeg_generator(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/snapshot')
        def snapshot():
            """Single JPEG frame."""
            frame_bytes = self._read_frame()
            if frame_bytes is None:
                return Response('Stream not ready', status=503)
            return Response(frame_bytes, mimetype='image/jpeg')

    def _mjpeg_generator(self):
        """Yield multipart/x-mixed-replace MJPEG chunks for /stream."""
        while True:
            frame_bytes = self._read_frame()
            if frame_bytes is None:
                # No frame produced yet; wait briefly rather than busy-spinning.
                time.sleep(0.05)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def start(self):
        """Launch the Flask dev server in a daemon thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self.app.run,
            kwargs={'host': self.host, 'port': self.port,
                    'threaded': True, 'use_reloader': False},
            daemon=True,
            name='birds-cv-stream',
        )
        self._thread.start()
        print(f'Serving live annotated stream on http://{self.host}:{self.port}/')

    def stop(self):
        """No-op: the Flask thread is a daemon and dies with the process."""
        pass
