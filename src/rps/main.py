import cv2

from .fps import FPSCounter
from .hand_tracker import HandTracker
from .viz import draw_hand


def main() -> None:
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

                                    
            if result.hand_landmarks:
                for i, hand_lms in enumerate(result.hand_landmarks):
                    draw_hand(frame, hand_lms)

                    if result.handedness and i < len(result.handedness) and result.handedness[i]:
                        cat = result.handedness[i][0]
                        label = f"{cat.category_name} ({cat.score:.2f})"
                        cv2.putText(
                            frame, label, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                        )

            fps = fps_counter.tick()
            cv2.putText(
                frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            cv2.imshow("RPS Vision - HandLandmarker (Tasks)", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()