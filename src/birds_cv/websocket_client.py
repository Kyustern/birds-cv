"""Socket.IO client for streaming per-frame detections to a remote server.

The emitter uses the synchronous ``python-socketio`` ``Client``, which runs its
own background thread for receiving and exposes a thread-safe ``emit()``. This
slots into the synchronous inference loop in :mod:`birds_cv.pipeline` without an
asyncio event loop. It is fault-tolerant: if the library is missing or the
server is unreachable it prints a warning and the inference loop continues
uninterrupted.
"""

import time
from urllib.parse import urlsplit

try:
    import socketio
except ImportError:
    socketio = None


def _to_socketio_url(url):
    """Normalize a URL for the Socket.IO client.

    ``python-socketio`` connects to an HTTP(S) origin and negotiates the
    WebSocket transport during the handshake, so a ``ws://``/``wss://`` URL is
    rewritten to ``http://``/``https://``. The path is returned separately so it
    can be used as the Socket.IO namespace (defaulting to ``/`` when empty).
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme == 'ws':
        scheme = 'http'
    elif scheme == 'wss':
        scheme = 'https'
    elif scheme not in ('http', 'https', ''):
        scheme = 'http'
    base = f'{scheme}://{parts.netloc}' if parts.netloc else url
    namespace = parts.path or '/'
    if not namespace.startswith('/'):
        namespace = '/' + namespace
    # A bare "/" path is the default namespace; anything else is left as-is so
    # the server can register it explicitly (e.g. "/api/ws").
    return base, namespace


class WebSocketEmitter:
    """Connect to a Socket.IO server and emit a 'detections' event per frame.

    Each emitted message is a JSON object of the form::

        {
            "frame": <int>,
            "timestamp": <float epoch seconds>,
            "detections": [
                {
                    "classname": <str>,
                    "conf": <float>,
                    "cls": <int>,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy]
                },
                ...
            ]
        }

    The message is emitted as the ``detections`` Socket.IO event in the
    namespace derived from the connection URL (default ``/``).
    """

    def __init__(self, url, timeout=5, reconnect=True):
        """
        Args:
            url: Socket.IO server URL (e.g. 'http://host:port' or
                'ws://host:port/api/ws'). A ``ws://``/``wss://`` scheme is
                rewritten to ``http://``/``https://`` automatically.
            timeout: Connection timeout in seconds.
            reconnect: If True, rely on the client's built-in reconnection
                logic after the connection drops.
        """
        self.url = url
        self.base_url, self.namespace = _to_socketio_url(url)
        self.timeout = timeout
        self.reconnect = reconnect
        self.sio = None
        self._available = socketio is not None

    def connect(self):
        """Open the Socket.IO connection. Returns True on success."""
        if not self._available:
            print('WARNING: python-socketio is not installed; '
                  'Socket.IO emission disabled. Install with: pip install python-socketio')
            return False
        try:
            self.sio = socketio.Client(
                reconnection=self.reconnect,
                reconnection_attempts=0 if self.reconnect else 1,
                reconnection_delay=1,
                reconnection_delay_max=5,
            )
            print("self.namespace", self.namespace)
            print("self.url", self.url)
            self.sio.connect(
                url=self.url,
                socketio_path="/api/ws",
                # namespaces=[self.namespace],
                wait_timeout=self.timeout,
            )
            print(f'Socket.IO connected to {self.base_url} (namespace {self.namespace})')
            return True
        except Exception as e:
            print(f'WARNING: could not connect Socket.IO to {self.url}: {e}')
            self.sio = None
            return False

    def emit_detections(self, detections, frame_count):
        """Emit a 'detections' event for the given frame.

        Silently no-ops when the client is not connected. The underlying client
        handles reconnection in its background thread, so a transient drop does
        not require an explicit reconnect attempt here; a failed emit simply
        drops the frame.
        """
        if self.sio is None:
            if self.reconnect:
                self.connect()
            if self.sio is None:
                return

        payload = {
            'frame': frame_count,
            'timestamp': time.time(),
            'detections': [self._serialize(d) for d in detections],
        }
        print("payload", payload)
        try:
            self.sio.emit('detections', payload)
        except Exception as e:
            print(f'WARNING: Socket.IO emit failed: {e}')

    @staticmethod
    def _serialize(det):
        """Convert a detection dict into a JSON-safe payload with bbox and center."""
        x1, y1, x2, y2 = det['xyxy'].tolist()
        return {
            'classname': det['classname'],
            'conf': det['conf'],
            'cls': det['cls'],
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
        }

    def close(self):
        """Close the Socket.IO connection if open."""
        if self.sio is not None:
            try:
                self.sio.disconnect()
            except Exception:
                pass
            self.sio = None
