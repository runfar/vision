import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread(cv.samples.findFile("test.png"))
h = cv.calcHist([img], [2], None, [256], [0, 256])
plt.plot(h, color = 'r', linewidth=2)
plt.show()