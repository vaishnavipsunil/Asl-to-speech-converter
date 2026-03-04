import cv2
import mediapipe as mp
import pandas as pd
import os
import string
import numpy as np

# ==========================
SAMPLES_PER_CLASS = 200
DATA_FILE = "sign_landmarks.csv"
# ==========================

labels = list(string.ascii_uppercase)  # A-Z
current_index = 0
current_label = labels[current_index]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

# Create CSV if not exists
if not os.path.exists(DATA_FILE):
    columns = []
    for i in range(21):
        columns += [f"x{i}", f"y{i}", f"z{i}"]
    columns.append("label")
    pd.DataFrame(columns=columns).to_csv(DATA_FILE, index=False)

cap = cv2.VideoCapture(0)

count = 0
collecting = False

print("Press 's' to start collecting")
print("Press 'n' to move to next letter")
print("Press 'q' to quit")

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if collecting and count < SAMPLES_PER_CLASS:

                # ==========================
                # PROFESSIONAL NORMALIZATION
                # ==========================

                wrist = hand_landmarks.landmark[0]
                middle_tip = hand_landmarks.landmark[12]

                # Calculate hand size (scale)
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

                row.append(current_label)

                pd.DataFrame([row]).to_csv(DATA_FILE, mode='a', header=False, index=False)

                count += 1
                print(f"{current_label} -> {count}/{SAMPLES_PER_CLASS}")

    # Display current letter & count
    cv2.putText(frame, f"Letter: {current_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Samples: {count}/{SAMPLES_PER_CLASS}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("ASL Data Collection - Professional", frame)

    key = cv2.waitKey(1) & 0xFF

    # Start collecting
    if key == ord('s'):
        collecting = True

    # Move to next letter
    if key == ord('n'):
        collecting = False
        count = 0
        current_index += 1

        if current_index < len(labels):
            current_label = labels[current_index]
        else:
            print("All letters collected!")
            break

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()