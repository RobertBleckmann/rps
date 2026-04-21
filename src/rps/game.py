import time
import random
from dataclasses import dataclass
from collections import Counter, deque

import cv2
import numpy as np
import joblib

from .config import DATA_DIR
from .hand_tracker import HandTracker
from .fps import FPSCounter
from .features import FeatureConfig, landmarks_to_feature_vector
from .viz import draw_hand


MODEL_FILE = DATA_DIR / "models" / "rps_model.joblib"

GESTURES = ["Rock", "Paper", "Scissors"]
WIN_MAP = {
    ("Rock", "Scissors"): 1,
    ("Scissors", "Paper"): 1,
    ("Paper", "Rock"): 1,
}


                               
            
                               

@dataclass
class Button:
    x: int
    y: int
    w: int
    h: int
    text: str

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


def put_text_centered(img, text, center_xy, font, scale, color, thickness, y_offset=0):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cx, cy = center_xy
    x = int(cx - tw / 2)
    y = int(cy + th / 2) + int(y_offset)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def put_text_left(img, text, x, y, font, scale, color, thickness):
    cv2.putText(img, text, (int(x), int(y)), font, scale, color, thickness, cv2.LINE_AA)


class GameUI:
    def __init__(self):
                
        self.W, self.H = 1280, 720

                           
        self.start_btn = Button(x=30, y=15, w=220, h=45, text="Start (Space)")
        self.reset_btn = Button(x=self.W - 200, y=15, w=170, h=45, text="Reset (R)")

                          
        self.top_w, self.top_h = 900, 420
        self.top_x = (self.W - self.top_w) // 2
        self.top_y = 70

                                  
        self.score_y = self.top_y + self.top_h + 45

                           
        self.result_y1 = self.score_y + 35
        self.result_y2 = self.H - 35

                           
        self._clicked_reset = False
        self._clicked_start = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.reset_btn.contains(x, y):
                self._clicked_reset = True
            if self.start_btn.contains(x, y):
                self._clicked_start = True

    def consume_reset_click(self) -> bool:
        if self._clicked_reset:
            self._clicked_reset = False
            return True
        return False

    def consume_start_click(self) -> bool:
        if self._clicked_start:
            self._clicked_start = False
            return True
        return False

    def make_canvas(self):
        return np.zeros((self.H, self.W, 3), dtype=np.uint8)

    def draw_button(self, canvas, btn: Button):
        cv2.rectangle(canvas, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), (60, 60, 60), -1)
        cv2.rectangle(canvas, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), (140, 140, 140), 2)
        put_text_left(canvas, btn.text, btn.x + 10, btn.y + 30,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def place_image_letterboxed(self, canvas, img_bgr, x, y, w, h):
        """
        Place image without distortion and return the inner placement rect:
        (ox, oy, new_w, new_h) for the actual pasted image area.
        """
        ih, iw = img_bgr.shape[:2]
        if iw <= 0 or ih <= 0:
            return None

        scale = min(w / iw, h / ih)
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

                                
        canvas[y:y + h, x:x + w] = 0

                      
        ox = x + (w - new_w) // 2
        oy = y + (h - new_h) // 2
        canvas[oy:oy + new_h, ox:ox + new_w] = resized

        return ox, oy, new_w, new_h


                               
            
                               

def decide_winner(player: str, cpu: str) -> int:
    if player == cpu:
        return 0
    if (player, cpu) in WIN_MAP:
        return 1
    return -1


def winner_line(player: str, cpu: str) -> str:
    res = decide_winner(player, cpu)
    if res == 1:
        return f"{player} wins"
    if res == -1:
        return f"{cpu} wins"
    return "Draw"


def main():
    if not MODEL_FILE.exists():
        print(f"Model not found: {MODEL_FILE}")
        print("Train first: python -m src.rps.train")
        return

    clf = joblib.load(MODEL_FILE)
    cfg = FeatureConfig(use_z=False)

    fps_counter = FPSCounter()
    ui = GameUI()

                
    state = "WAITING"                                                       
    countdown_start = 0.0

    score_player = 0
    score_cpu = 0

    locked_player = None
    locked_cpu = None
    locked_conf = 0.0

                                                        
    pred_buffer = deque(maxlen=30)

                                                          
    round_matchup = ""                            
    round_outcome = ""                     

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    window_name = "RPS Game"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, ui.mouse_callback)

    with HandTracker() as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

                         
            result = tracker.detect(frame)

                             
            raw_label = None
            conf = 0.0

            if result.hand_landmarks:
                hand_lms = result.hand_landmarks[0]
                feat = landmarks_to_feature_vector(hand_lms, cfg).reshape(1, -1)
                probs = clf.predict_proba(feat)[0]
                classes = clf.classes_
                best_idx = int(probs.argmax())
                raw_label = str(classes[best_idx])
                conf = float(probs[best_idx])

                                  
            canvas = ui.make_canvas()

                     
            ui.draw_button(canvas, ui.start_btn)
            ui.draw_button(canvas, ui.reset_btn)

                          
            placement = ui.place_image_letterboxed(canvas, frame, ui.top_x, ui.top_y, ui.top_w, ui.top_h)

                                               
            webcam_center_x = ui.top_x + ui.top_w // 2
            put_text_centered(
                canvas, "Webcam",
                (webcam_center_x, ui.top_y - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )

                                                                                                     
            if placement is not None:
                ox, oy, new_w, new_h = placement
                view = canvas[oy:oy + new_h, ox:ox + new_w]                                      

                if result.hand_landmarks:
                                                             
                    draw_hand(view, result.hand_landmarks[0])

                                            
                if raw_label in GESTURES:
                    put_text_left(
                        view,
                        f"Live: {raw_label} ({conf:.2f})",
                        12, 32,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (0, 255, 255), 2
                    )
                else:
                    put_text_left(
                        view,
                        "Live: No hand",
                        12, 32,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (180, 180, 180), 2
                    )

                                                                               
            score_text = f"Player: {score_player}    CPU: {score_cpu}"
            put_text_centered(
                canvas, score_text,
                (webcam_center_x, ui.score_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
            )

                          
            key = cv2.waitKey(1) & 0xFF
            if key == 27:       
                break

                   
            if ui.consume_reset_click() or key in (ord("r"), ord("R")):
                score_player = 0
                score_cpu = 0
                state = "WAITING"
                pred_buffer.clear()
                locked_player = None
                locked_cpu = None
                locked_conf = 0.0
                round_matchup = ""
                round_outcome = ""

                                
            start_pressed = ui.consume_start_click() or key == ord(" ")
            if state in ("WAITING", "RESULT") and start_pressed:
                state = "COUNTDOWN"
                countdown_start = time.time()
                pred_buffer.clear()

                             
            countdown_value = None
            if state == "COUNTDOWN":
                elapsed = time.time() - countdown_start
                if elapsed < 1:
                    countdown_value = 3
                elif elapsed < 2:
                    countdown_value = 2
                elif elapsed < 3:
                    countdown_value = 1
                else:
                    countdown_value = 0

                if raw_label in GESTURES:
                    pred_buffer.append((raw_label, conf))

                if elapsed >= 3.0:
                                 
                    if len(pred_buffer) > 0:
                        labels = [p[0] for p in pred_buffer]
                        locked_player = Counter(labels).most_common(1)[0][0]
                        confs = [p[1] for p in pred_buffer if p[0] == locked_player]
                        locked_conf = float(sum(confs) / max(len(confs), 1))
                    else:
                        locked_player = "Rock"
                        locked_conf = 0.0

                                
                    locked_cpu = random.choice(GESTURES)

                             
                    res = decide_winner(locked_player, locked_cpu)
                    if res == 1:
                        score_player += 1
                    elif res == -1:
                        score_cpu += 1

                                                
                    round_matchup = f"{locked_player} vs {locked_cpu}"
                    round_outcome = winner_line(locked_player, locked_cpu)

                    state = "RESULT"
                    pred_buffer.clear()

                                                            
            if state == "COUNTDOWN" and countdown_value is not None:
                put_text_centered(
                    canvas, str(countdown_value),
                    (ui.top_x + ui.top_w // 2, ui.top_y + ui.top_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 255), 8, y_offset=30
                )

                                                                 
            if round_matchup:
                put_text_centered(
                    canvas, round_matchup,
                    (ui.W // 2, ui.score_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2
                )
                put_text_centered(
                    canvas, round_outcome,
                    (ui.W // 2, ui.score_y + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
                )

                              
            fps = fps_counter.tick()
            put_text_left(canvas, f"FPS: {int(fps)}", ui.W - 140, ui.H - 20,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window_name, canvas)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()