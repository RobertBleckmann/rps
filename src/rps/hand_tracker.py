import time
from dataclasses import dataclass
import cv2
import mediapipe as mp

from .config import MODEL_PATH                                                 


@dataclass(frozen=True)
class HandTrackerConfig:
    num_hands: int = 1
    min_hand_detection_confidence: float = 0.7
    min_hand_presence_confidence: float = 0.7
    min_tracking_confidence: float = 0.7


class HandTracker:
    """Synchronous hand landmark tracker using MediaPipe Tasks (VIDEO mode)."""

    def __init__(self, cfg: HandTrackerConfig = HandTrackerConfig()):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self._start_time = time.time()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_hand_detection_confidence,
            min_hand_presence_confidence=cfg.min_hand_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

        self._landmarker_cm = HandLandmarker.create_from_options(options)
        self._landmarker = None

    def __enter__(self):
                                                  
        self._landmarker = self._landmarker_cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._landmarker_cm.__exit__(exc_type, exc, tb)

    def detect(self, frame_bgr):
        """Returns result from detect_for_video."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)