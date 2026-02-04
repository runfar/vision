import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("test.png"))

if img is None:
    sys.exit("Could not read the image.")


def draw(event, x, y, flags, param):
    global ix, iy
    if event == cv.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    elif event == cv.EVENT_LBUTTONUP:
        cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

    cv.imshow("Image with Mouse Events", img)


cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw)

while(True):
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break