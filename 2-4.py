import cv2 as cv
import sys

# Windows용 CAP_DSHOW 대신 macOS에서는 CAP_AVFOUNDATION 사용 또는 인자 생략
cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)

if not cap.isOpened():
    sys.exit("Could not open camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('Live Video - Press q to Exit', gray)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()