import cv2 as cv
import numpy as np
import sys
from PyQt5.QtWidgets import *

class Orim(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('오림')
        self.setGeometry(200, 200, 700, 100)

        fileButton = QPushButton('파일 열기', self)
        paintButton = QPushButton('페인팅', self)
        cutButton = QPushButton('오림', self)
        incButton = QPushButton('+', self)
        decButton = QPushButton('-', self)
        saveButton = QPushButton('저장', self)
        quitButton = QPushButton('나가기', self)

        fileButton.setGeometry(10, 10, 100, 30)
        paintButton.setGeometry(110, 10, 100, 30)
        cutButton.setGeometry(210, 10, 100, 30)
        incButton.setGeometry(310, 10, 50, 30)
        decButton.setGeometry(370, 10, 50, 30)
        saveButton.setGeometry(430, 10, 100, 30)
        quitButton.setGeometry(530, 10, 100, 30)

        fileButton.clicked.connect(self.openFile)
        paintButton.clicked.connect(self.paint)
        cutButton.clicked.connect(self.cut)
        incButton.clicked.connect(self.increase)
        decButton.clicked.connect(self.decrease)
        saveButton.clicked.connect(self.save)
        quitButton.clicked.connect(self.quit)

        self.BrushSize = 5
        self.LColor, self.RColor = (255, 0, 0), (0, 0, 255)

    def openFile(self):
        fname = QFileDialog.getOpenFileName(self, '파일 열기', './')
        print(fname)
        self.img = cv.imread(fname[0])
        self.img_show = np.copy(self.img)
        cv.imshow('Original Image', self.img)    

        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        self.mask[:,:] = cv.GC_PR_BGD

    def paint(self):
        cv.setMouseCallback('Original Image', self.painting)        

    def painting(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.BrushSize, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSize, cv.GC_FGD, -1)
        elif event == cv.EVENT_RBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.BrushSize, self.RColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSize, cv.GC_BGD, -1)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            cv.circle(self.img_show, (x, y), self.BrushSize, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSize, cv.GC_FGD, -1)
        cv.imshow('Painting', self.img_show)

    def cut(self):
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (1, 1, self.img.shape[1]-2, self.img.shape[0]-2)

        cv.grabCut(self.img, self.mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
        mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        self.result = self.img * mask2[:, :, np.newaxis]
        cv.imshow('Cut Image', self.result)

    def increase(self):
        self.BrushSize += 1
        print("Brush Size:", self.BrushSize)

    def decrease(self):
        self.BrushSize = max(1, self.BrushSize - 1)
        print("Brush Size:", self.BrushSize)

    def save(self):
        fname = QFileDialog.getSaveFileName(self, '파일 저장', './')
        print(fname)
        cv.imwrite(fname[0], self.result)

    def quit(self):
        cv.destroyAllWindows()
        self.close()
                                                

app = QApplication(sys.argv)
win = Orim()
win.show()
app.exec_()