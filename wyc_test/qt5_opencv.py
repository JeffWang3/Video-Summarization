from PyQt5 import QtGui, QtCore, QtWidgets
import cv2
import sys

import cv2
import numpy as np
from tqdm import tqdm
import os

class DisplayImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DisplayImageWidget, self).__init__(parent)

        self.button = QtWidgets.QPushButton('Show picture')
        self.button.clicked.connect(self.show_image)
        self.image_frame = QtWidgets.QLabel()

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.image_frame)
        self.setLayout(self.layout)

    @QtCore.pyqtSlot()
    def show_image(self):
        images = self.getFrameArray()
        for filename in tqdm(images):
            self.image = cv2.imread(filename)
            self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))

    def getFrameArray(self, folderpath='C:\\Users\\16200\\Desktop\\project_dataset\\frames\\soccer'):
        path = folderpath
        img_array = []
        namelist = sorted(os.listdir(path), key = lambda x: int(x[5:-4]))
        return namelist
        # for filename in tqdm(namelist):
        #     imgpath = os.path.join(path, filename)
        #     img = cv2.imread(imgpath)
        #     height, width, layers = img.shape
        #     # size = (width,height)
        #     img_array.append(img)
        # return img_array   

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    display_image_widget = DisplayImageWidget()
    display_image_widget.show()
    sys.exit(app.exec_())