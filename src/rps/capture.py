import csv
from pathlib import Path

import cv2

from .config import DATA_DIR
from .features import FeatureConfig, landmarks_to_feature_vector
from .fps import FPSCounter
from .hand_tracker import HandTracker
from .viz import draw_hand


DATA_PATH = DATA_DIR / "samples.csv"


def ensure_header(path: Path, feature_len: int) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["label"] + [f"f{i}" for i in range(feature_len)]
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def append_row(path: Path, label: str, feat) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([label] + list(map(float, feat)))


def _dummy_feature_len(cfg: FeatureConfig) -> int:
                                                                                              
    DummyLM = type("LM", (), {"x": 0.0, "y": 0.0, "z": 0.0})
    dummy_hand = [DummyLM() for _ in range(21)]
    return len(landmarks_to_feature_vector(dummy_hand, cfg))


def main() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    cfg = FeatureConfig(use_z=False)
    ensure_header(DATA_PATH, _dummy_feature_len(cfg))

    fps_counter = FPSCounter()

    print("Recording controls: r=rock, p=paper, s=scissors, ESC=quit")

    with HandTracker() as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = tracker.detect(frame)

            label_hint = "No hand"
            feats = None

            if result.hand_landmarks:
                hand_lms = result.hand_landmarks[0]
                draw_hand(frame, hand_lms)
                feats = landmarks_to_feature_vector(hand_lms, cfg)
                label_hint = "Hand OK"

            fps = fps_counter.tick()

            cv2.putText(
                frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"{label_hint} | saving to: {DATA_PATH.name}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            cv2.imshow("RPS Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:       
                break

            if feats is None:
                continue                              

            if key in (ord("r"), ord("p"), ord("s")):
                label = chr(key)               
                append_row(DATA_PATH, label, feats)
                print(f"Saved sample: {label}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()