"""Start the Raspberry Pi camera and serve its live feed through a Flask API.

The camera is started with birds_cv.camera.start_rpicam_source(), which launches
rpicam-vid and exposes the MJPEG stream on a named pipe. This script opens that
pipe with OpenCV and re-emits the frames as a Motion JPEG stream served by Flask,
so any browser or HTTP client can consume the live feed at:

    http://<host>:5000/           -> HTML page embedding the feed
    http://<host>:5000/stream     -> raw multipart/x-mixed-replace MJPEG stream
    http://<host>:5000/snapshot   -> single JPEG frame
"""

import signal
import threading

import cv2
from flask import Flask, Response

from birds_cv import camera


HOST = '0.0.0.0'
PORT = 5010

app = Flask(__name__)

# Camera state, populated by start_camera() before Flask begins serving requests.
_fifo_path = None
_rpicam_proc = None
_cap = None
_lock = threading.Lock()


def start_camera(width=640, height=480, framerate=30):
    """Start rpicam-vid and open the resulting MJPEG pipe with OpenCV.

    Returns True on success, False if the camera or pipe could not be opened.
    """
    global _fifo_path, _rpicam_proc, _cap
    _fifo_path, _rpicam_proc = camera.start_rpicam_source(
        width=width, height=height, framerate=framerate
    )
    if _fifo_path is None:
        return False

    _cap = cv2.VideoCapture(_fifo_path)
    if not _cap.isOpened():
        print(f'ERROR: could not open camera pipe {_fifo_path}')
        stop_camera()
        return False
    return True


def stop_camera():
    """Release the OpenCV capture and stop the rpicam-vid subprocess."""
    global _cap, _rpicam_proc, _fifo_path
    if _cap is not None:
        _cap.release()
        _cap = None
    camera.stop_rpicam_source(_rpicam_proc)
    _rpicam_proc = None
    _fifo_path = None


def _read_frame():
    """Read a single frame from the camera under the stream lock. Returns bytes or None."""
    with _lock:
        if _cap is None or not _cap.isOpened():
            return None
        ret, frame = _cap.read()
        if not ret:
            return None
        ok, encoded = cv2.imencode('.jpg', frame)
        if not ok:
            return None
        return encoded.tobytes()


def mjpeg_generator():
    """Yield multipart/x-mixed-replace MJPEG chunks for the /stream endpoint."""
    while True:
        frame_bytes = _read_frame()
        if frame_bytes is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


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
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot')
def snapshot():
    """Single JPEG frame."""
    frame_bytes = _read_frame()
    if frame_bytes is None:
        return Response('Camera unavailable', status=503)
    return Response(frame_bytes, mimetype='image/jpeg')


def main():
    if not start_camera():
        print('ERROR: could not start camera; aborting.')
        return

    def _shutdown(_signum, _frame):
        stop_camera()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f'Serving live camera stream on http://{HOST}:{PORT}/')
    try:
        # threaded=True lets /stream and /snapshot be served concurrently.
        app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
    finally:
        stop_camera()


if __name__ == '__main__':
    main()