"""Live sign -> text -> speech. Press Q to quit."""
import pickle, threading, time
from collections import deque, Counter
import cv2
import mediapipe as mp
import pyttsx3

model = pickle.load(open("model.pkl", "rb"))
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils


def features(hand):
    pts = [(p.x, p.y, p.z) for p in hand.landmark]
    wx, wy, wz = pts[0]
    rel = [(x - wx, y - wy, z - wz) for x, y, z in pts]
    m = max(max(abs(a) for a in p) for p in rel) or 1
    return [v / m for p in rel for v in p]


def speak(text):
    def _run():
        engine = pyttsx3.init()
        engine.say(text.replace("_", " ").lower())
        engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()


recent = deque(maxlen=10)   # smoothing: majority vote over last 10 frames
last_spoken, last_time = None, 0
shown = ""

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        h = res.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, h, mp.solutions.hands.HAND_CONNECTIONS)
        probs = model.predict_proba([features(h)])[0]
        if probs.max() > 0.7:  # ignore low-confidence guesses
            recent.append(model.classes_[probs.argmax()])
        if len(recent) == recent.maxlen:
            label, n = Counter(recent).most_common(1)[0]
            if n >= 8:
                shown = label
                if label != last_spoken and time.time() - last_time > 2:
                    speak(label)
                    last_spoken, last_time = label, time.time()
    else:
        recent.clear()
        last_spoken = None

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Text: {shown}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
