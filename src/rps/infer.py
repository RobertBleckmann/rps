from pathlib import Path

import cv2
import joblib

from .config import DATA_DIR
from .features import FeatureConfig, landmarks_to_feature_vector
from .fps import FPSCounter
from .hand_tracker import HandTracker
from .viz import draw_hand
from .smoothing import MajorityVoteSmoother


MODEL_FILE = DATA_DIR / "models" / "rps_model.joblib"


def main() -> None:
    if not MODEL_FILE.exists():
        print(f"Model not found: {MODEL_FILE}")
        print("Train first: python -m src.rps.train")
        return

    clf = joblib.load(MODEL_FILE)
    smoother = MajorityVoteSmoother(window_size=9)
    cfg = FeatureConfig(use_z=False)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    fps_counter = FPSCounter()

    with HandTracker() as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = tracker.detect(frame)

            label = "No hand"
            conf = 0.0

            if result.hand_landmarks:
                hand_lms = result.hand_landmarks[0]
                draw_hand(frame, hand_lms)

                feat = landmarks_to_feature_vector(hand_lms, cfg).reshape(1, -1)

                                                              
                probs = clf.predict_proba(feat)[0]
                classes = clf.classes_
                best_idx = int(probs.argmax())
                raw_label = str(classes[best_idx])
                conf = float(probs[best_idx])

                                            
                label = smoother.update(raw_label)

            fps = fps_counter.tick()

            cv2.putText(
                frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Pose: {label} ({conf:.2f})", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            cv2.imshow("RPS Infer", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()