import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("test.png"))

if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Original Image", img)
cv.imshow("Upper left half", img[0:img.shape[0]//2, 0:img.shape[1]//2])
cv.imshow("Center half", img[img.shape[0]//4:3*img.shape[0]//4, 
                             img.shape[1]//4:3*img.shape[1]//4])

print("Image shape:", img.shape)

cv.imshow("R channel", img[:, :, 2])
cv.imshow("G channel", img[:, :, 1])
cv.imshow("B channel", img[:, :, 0])

cv.waitKey(0)
cv.destroyAllWindows()