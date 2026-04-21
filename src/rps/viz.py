import cv2

                  
'''
4   8   12  16  20
|   |   |   |   |
3   7   11  15  19
|   |   |   |   |
2   6   10  14  18
|   |   |   |   |
1   5 - 9 - 13 - 17
|   |   -   -   |
        |      
        0 (Wrist)
'''
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                
    (0, 5), (5, 6), (6, 7), (7, 8),                
    (5, 9), (9, 10), (10, 11), (11, 12),            
    (9, 13), (13, 14), (14, 15), (15, 16),        
    (13, 17), (17, 18), (18, 19), (19, 20),        
    (0, 17)                                            
]


def draw_hand(frame, hand_landmarks, color=(0, 255, 0)) -> None:
    h, w = frame.shape[:2]

            
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 4, color, -1)

           
    for start_idx, end_idx in HAND_CONNECTIONS:
        x1 = int(hand_landmarks[start_idx].x * w)
        y1 = int(hand_landmarks[start_idx].y * h)
        x2 = int(hand_landmarks[end_idx].x * w)
        y2 = int(hand_landmarks[end_idx].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)