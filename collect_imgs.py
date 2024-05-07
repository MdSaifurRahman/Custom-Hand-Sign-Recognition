import os
import cv2 as cv

DIRECTORY = "./data"
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

no_of_labels = 26
size = 100

capture = cv.VideoCapture(0)
for i in range(no_of_labels):
    if not os.path.exists(os.path.join(DIRECTORY, str(i))):
        os.makedirs(os.path.join(DIRECTORY, str(i)))

    print("Collecting data for class {}".format(i))

    done = False
    while True:
        ret, frame = capture.read()
        cv.putText(
            frame,
            'Press "D" to Start ! :)',
            (100, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3,
            cv.LINE_AA,
        )
        cv.imshow("Live", frame)
        if cv.waitKey(25) == ord("d"):
            break

    c = 0
    while c < size:
        ret, frame = capture.read()
        cv.imshow("Live", frame)
        cv.waitKey(25)
        cv.imwrite(os.path.join(DIRECTORY, str(i), "{}.jpg".format(c)), frame)

        c += 1

capture.release()
cv.destroyAllWindows()
