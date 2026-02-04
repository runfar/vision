import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("test.png"))


if img is None:
    sys.exit("Could not read the image.")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_small = cv.resize(gray, dsize=(0, 0), fx=0.5, fy=0.5)

cv.imwrite("gray.png", gray)
cv.imwrite("gray_small.png", gray_small)

cv.imshow("Original Image", img)
cv.imshow("Gray Image", gray)
cv.imshow("Gray Small Image", gray_small)

cv.waitKey(0)
cv.destroyAllWindows()