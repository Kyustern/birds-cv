"""Raspberry Pi camera source via the libcamera (rpicam-vid) stack."""

import os
import atexit
import subprocess
import tempfile


def start_rpicam_source(width=640, height=480, framerate=30):
    """
    Start the Raspberry Pi camera via the libcamera stack (rpicam-vid) and expose
    the live feed as an MJPEG stream on a named pipe that cv2.VideoCapture can read.

    On modern Raspberry Pi OS (Bullseye+/Trixie) the camera is served by libcamera,
    so /dev/video0 cannot be opened directly by OpenCV -- rpicam-vid configures the
    sensor/ISP pipeline and writes MJPEG frames to a FIFO instead.

    Returns (fifo_path, process) or (None, None) on failure.
    """
    tmpdir = tempfile.mkdtemp(prefix='rpicam_vibe_')
    fifo_path = os.path.join(tmpdir, 'stream.mjpeg')
    os.mkfifo(fifo_path)
    cmd = [
        'rpicam-vid', '-t', '0',
        '--width', str(width), '--height', str(height),
        '--framerate', str(framerate),
        '--codec', 'mjpeg', '-o', fifo_path,
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print('ERROR: rpicam-vid not found. On Raspberry Pi OS install it with: '
              'sudo apt install rpicam-apps')
        return None, None

    def _cleanup(_proc=proc):
        if _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _proc.kill()
    atexit.register(_cleanup)

    print(f'Started rpicam-vid (pid {proc.pid}) streaming MJPEG to {fifo_path}')
    return fifo_path, proc


def stop_rpicam_source(proc):
    """Terminate an rpicam-vid subprocess started by start_rpicam_source()."""
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
