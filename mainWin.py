from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(625, 430)

        # set image display widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.ImgDisp = QtWidgets.QLabel(self.centralwidget)
        self.ImgDisp.setGeometry(QtCore.QRect(0, 0, 54, 12))
        self.ImgDisp.setObjectName("ImgDisplay")
        MainWindow.setCentralWidget(self.centralwidget)

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

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ImgDisp.setText(_translate("MainWindow", "."))
        self.menushowImg.setTitle(_translate("MainWindow", "menu"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionshowImg.setText(_translate("MainWindow", "showImg"))
        self.actionPlayOrPause.setText(_translate("MainWindow", "PlayOrPause"))