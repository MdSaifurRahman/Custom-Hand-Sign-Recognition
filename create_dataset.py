import os
import pickle
import mediapipe as mp
import cv2 as cv
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DIRECTORY = "./data"

data = []
labels = []
for files in os.listdir(DIRECTORY):
    for img_path in os.listdir(os.path.join(DIRECTORY, files)):
        auxillary_data = []

        xpoints = []
        ypoints = []

        img = cv.imread(os.path.join(DIRECTORY, files, img_path))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
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

            data.append(auxillary_data)
            labels.append(files)

f = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()
