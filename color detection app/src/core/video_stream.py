"""
Thread-safe Asynchronous Video Stream Engine.
Decouples camera frame ingestion from computer vision processing and GUI rendering,
preventing UI freezing and pipeline latency.
"""

import threading
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from config.settings import VISION_CONFIG
from src.utils.logger import app_logger


class VideoStream:
    """
    Asynchronous threaded video reader supporting webcams and video files.
    """

    def __init__(self, src: Union[int, str] = VISION_CONFIG.CAMERA_INDEX):
        self.src = src
        self.stream: Optional[cv2.VideoCapture] = None
        self.grabbed: bool = False
        self.frame: Optional[np.ndarray] = None
        self.stopped: bool = True
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        # Real-time FPS monitoring
        self.fps: float = 0.0
        self._prev_time: float = time.time()
        self._frame_count: int = 0

    def start(self) -> "VideoStream":
        """Initializes hardware capture and launches the background reader thread."""
        if not self.stopped:
            return self

        app_logger.info(f"Connecting to video stream source: {self.src}")
        
        # On Windows, cv2.CAP_DSHOW provides fast camera initialization
        if isinstance(self.src, int):
            self.stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        else:
            self.stream = cv2.VideoCapture(self.src)

        if not self.stream.isOpened():
            # Fallback to default backend
            self.stream = cv2.VideoCapture(self.src)

        if not self.stream.isOpened():
            app_logger.error(f"Failed to open video source: {self.src}")
            self.stopped = True
            return self

        # Set resolution
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, VISION_CONFIG.FRAME_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, VISION_CONFIG.FRAME_HEIGHT)

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

        self.thread = threading.Thread(target=self._update, name="VideoStreamThread", daemon=True)
        self.thread.start()
        app_logger.info("Video stream thread started successfully.")
        return self

    def _update(self) -> None:
        """Continuously reads latest frames in a dedicated thread."""
        while not self.stopped:
            if not self.stream or not self.stream.isOpened():
                break

            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stopped = True
                break

            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

            # FPS calculation
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._prev_time
            if elapsed >= 0.5:
                self.fps = round(self._frame_count / elapsed, 1)
                self._frame_count = 0
                self._prev_time = now

            # Throttle if needed to prevent 100% CPU spinning
            time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Thread-safe retrieval of the most recent frame.
        
        Returns:
            Tuple of (success_boolean, bgr_frame_array).
        """
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def get_fps(self) -> float:
        """Returns the current smoothed streaming FPS."""
        return self.fps

    def stop(self) -> None:
        """Gracefully stops background capture and releases camera hardware."""
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.stream and self.stream.isOpened():
            self.stream.release()
            app_logger.info("Video source released.")

    def __enter__(self) -> "VideoStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
