import pickle
import cv2 as cv
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]

capture = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

import string

char_labels = {i: letter.upper() for i, letter in enumerate(string.ascii_uppercase)}

while True:

    auxillary_data = []
    xpoints = []
    ypoints = []

    ret, frame = capture.read()

    H, W, _ = frame.shape

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                xpoints.append(x)
                ypoints.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                auxillary_data.append(x - min(xpoints))
                auxillary_data.append(y - min(ypoints))

        # Pad auxillary_data with zeros to ensure it has 100 features so we dont get shape errors
        auxillary_data += [0] * (100 - len(auxillary_data))

        x1 = int(min(xpoints) * W) - 10
        y1 = int(min(ypoints) * H) - 10

        x2 = int(max(xpoints) * W) - 10
        y2 = int(max(ypoints) * H) - 10

        pred = model.predict([np.asarray(auxillary_data)])

        prediction = char_labels[int(pred[0])]

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv.putText(
            frame,
            prediction,
            (x1, y1 - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            3,
            cv.LINE_AA,
        )

    cv.imshow("SIgn Language Predictor", frame)
    if cv.waitKey(20) & 0xFF == ord(
        "d"
    ):  # if letter d is pressed then the video will stop
        break
capture.release()
cv.destroyAllWindows()
