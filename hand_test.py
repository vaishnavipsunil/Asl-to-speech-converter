import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
from collections import deque, Counter
import pyttsx3
import time

# ========================
# LOAD MODEL & ENCODER
# ========================
model = tf.keras.models.load_model("asl_model.h5")

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ========================
# TEXT TO SPEECH
# ========================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ========================
# MEDIAPIPE HANDS
# ========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ========================
# VARIABLES
# ========================
prediction_buffer = deque(maxlen=10)
sentence = ""
last_time = 0
cooldown = 1.2 # Time gap between letters

# ========================
# MAIN LOOP
# ========================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_display_letter = ""
    current_conf = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # NORMALIZATION
            wrist = hand_landmarks.landmark[0]
            middle_tip = hand_landmarks.landmark[12]

            scale = np.sqrt(
                (middle_tip.x - wrist.x) ** 2 +
                (middle_tip.y - wrist.y) ** 2 +
                (middle_tip.z - wrist.z) ** 2
            )

            if scale == 0:
                continue

            row = []
            for lm in hand_landmarks.landmark:
                row.extend([
                    (lm.x - wrist.x) / scale,
                    (lm.y - wrist.y) / scale,
                    (lm.z - wrist.z) / scale
                ])

            X = np.array(row).reshape(1, -1)

            prediction = model.predict(X, verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > 0.85:
                letter = encoder.inverse_transform([class_id])[0]
                prediction_buffer.append(letter)

                if len(prediction_buffer) == 10:
                    most_common = Counter(prediction_buffer).most_common(1)[0][0]

                    current_display_letter = most_common
                    current_conf = confidence

                    current_time = time.time()

                    # ✅ ALLOW DOUBLE LETTERS AFTER COOLDOWN
                    if (current_time - last_time) > cooldown:
                        sentence += most_common
                        last_time = current_time

    # ========================
    # DISPLAY TEXT
    # ========================
    cv2.putText(frame,
                f"Letter: {current_display_letter} ({current_conf:.2f})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.putText(frame,
                f"Word: {sentence}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 0, 0),
                3)

    cv2.putText(frame,
                "S: Speak | C: Delete | X: Clear | Q: Quit",
                (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2)

    cv2.imshow("ASL Real-Time Prediction", frame)

    key = cv2.waitKey(1) & 0xFF

    # ========================
    # SPEAK
    # ========================
    if key == ord('s'):
        if sentence != "":
            engine.stop()
            engine.say(sentence)
            engine.runAndWait()

    # ========================
    # DELETE LAST LETTER
    # ========================
    if key == ord('c'):
        if len(sentence) > 0:
            sentence = sentence[:-1]

    # ========================
    # CLEAR FULL WORD
    # ========================
    if key == ord('x'):
        sentence = ""
        prediction_buffer.clear()

    # ========================
    # EXIT
    # ========================
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()