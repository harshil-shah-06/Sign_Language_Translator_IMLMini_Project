"""Collect training samples.
Usage:  python 1_collect.py HELLO
Hold the sign in front of the webcam. Press SPACE to start/stop recording,
Q to quit. Aim for ~200 samples per sign, moving the hand slightly while recording.
"""
import sys, csv, os
import cv2
import mediapipe as mp

label = sys.argv[1].upper()
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils


def features(hand):
    pts = [(p.x, p.y, p.z) for p in hand.landmark]
    wx, wy, wz = pts[0]
    rel = [(x - wx, y - wy, z - wz) for x, y, z in pts]
    m = max(max(abs(a) for a in p) for p in rel) or 1
    return [v / m for p in rel for v in p]  # 63 numbers, position/scale invariant


cap = cv2.VideoCapture(0)
recording, count = False, 0
new_file = not os.path.exists("data.csv")
f = open("data.csv", "a", newline="")
w = csv.writer(f)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        h = res.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, h, mp.solutions.hands.HAND_CONNECTIONS)
        if recording:
            w.writerow(features(h) + [label])
            count += 1
    status = "RECORDING" if recording else "paused"
    cv2.putText(frame, f"{label} | {status} | samples: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Collect", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        recording = not recording
    elif k == ord("q"):
        break

f.close()
cap.release()
cv2.destroyAllWindows()
