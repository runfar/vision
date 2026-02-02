import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("test.png"))

if img is None:
    sys.exit("Could not read the image.")


cv.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)
cv.circle(img, (300, 300), 50, (255, 0, 0), -1)
cv.line(img, (400, 100), (500, 200), (0, 0, 255), 5)
cv.putText(img, 'OpenCV', (100, 400), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

cv.imshow("Image with Shapes", img)
cv.waitKey(0)
cv.destroyAllWindows()