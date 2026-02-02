import cv2 as cv
import numpy as np
import sys

cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)

if not cap.isOpened():
    sys.exit("Could not open camera.")

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow('Live Video - Press s to Save Frame, q to Exit', frame)
    
    key = cv.waitKey(1)

    if key == ord('s'):
        frames.append(frame)
        print(f"Frame saved. Total saved frames: {len(frames)}")
    elif key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

if len(frames) > 0:
    imags = frames[0]
    for i in range(1, min(3, len(frames))):
        imags = np.hstack((imags, frames[i]))
    cv.imshow('Saved Frames', imags)

    cv.waitKey(0)
    cv.destroyAllWindows()