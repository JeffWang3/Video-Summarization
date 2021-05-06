from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QPushButton, QHBoxLayout, QVBoxLayout, QStyle


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(625, 430)

        self.playstate = False

        #set central Widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)

        # set image display widget
        self.ImgDispwidget = QtWidgets.QWidget(MainWindow)
        self.ImgDispwidget.setObjectName("ImgDispwidget")
        self.ImgDisp = QtWidgets.QLabel(self.ImgDispwidget)
        #self.ImgDisp.setGeometry(QtCore.QRect(0, 0, 54, 12))
        self.ImgDisp.setObjectName("ImgDisplay")
        #MainWindow.setCentralWidget(self.centralwidget)
        '''
        # set menu bar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 625, 17))
        self.menubar.setObjectName("menubar")
        self.menushowImg = QtWidgets.QMenu(self.menubar)
        self.menushowImg.setObjectName("menushowImg")
        MainWindow.setMenuBar(self.menubar)

        # set status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # set tool bar
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        # show image as an action
        self.actionshowImg = QtWidgets.QAction(MainWindow)
        self.actionshowImg.setObjectName("actionshowImg")
        self.menushowImg.addAction(self.actionshowImg)
        self.menubar.addAction(self.menushowImg.menuAction())
        self.toolBar.addAction(self.actionshowImg)

        # play or pause action
        self.actionPlayOrPause = QtWidgets.QAction(MainWindow)
        self.actionPlayOrPause.setObjectName("PlarOrPause")
        self.menushowImg.addAction(self.actionPlayOrPause)
        self.menubar.addAction(self.menushowImg.menuAction())
        self.toolBar.addAction(self.actionPlayOrPause)
        '''

        # summrize button
        self.summarizebtn = QPushButton('Summarize')
        self.summarizebtn.clicked.connect(self.iconchange)

        # open button
        self.openbtn = QPushButton('Replay')
        self.openbtn.setEnabled(False)

        # play or pause button
        self.playbtn = QPushButton()
        #self.playbtn.setEnabled(False)
        self.playbtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playbtn.clicked.connect(self.iconchange)

        # slide action
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setObjectName("Slider")
        self.slider.setRange(0,100)
        #self.menushowImg.addAction(self.actionPlayOrPause)
        #self.menubar.addAction(self.menushowImg.menuAction())
        #self.toolBar.addAction(self.actionPlayOrPause)

        # timer
        self.playtimer = QtWidgets.QLabel("00:00/_")
        #self.sp = QtWidgets.QSpinBox()


        #set layout
        hboxLayoutup = QHBoxLayout()
        hboxLayoutup.setContentsMargins(0,0,0,0);
        hboxLayoutup.addWidget(self.summarizebtn)
        hboxLayoutup.addWidget(self.openbtn)

        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0,0,0,0);
        hboxLayout.addWidget(self.playbtn)
        hboxLayout.addWidget(self.slider)
        hboxLayout.addWidget(self.playtimer)
        #hboxLayout.addWidget(self.sp)

        vboxlayout = QVBoxLayout()
        vboxlayout.addLayout(hboxLayoutup)
        vboxlayout.addWidget(self.ImgDispwidget)
        vboxlayout.addLayout(hboxLayout)

        self.centralwidget.setLayout(vboxlayout)

        '''
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        '''
    def iconchange(self):
        if self.playstate == False:
            self.playstate = True
            self.playbtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playstate = False
            self.playbtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionchange(self, position):
        self.slider.serValue(position)

    def durationchange(self, duration):
        self.slider.setRange(duration)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ImgDisp.setText(_translate("MainWindow", "."))
        self.menushowImg.setTitle(_translate("MainWindow", "menu"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionshowImg.setText(_translate("MainWindow", "showImg"))
        self.actionPlayOrPause.setText(_translate("MainWindow", "PlayOrPause"))