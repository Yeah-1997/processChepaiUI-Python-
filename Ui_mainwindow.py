# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'f:\pyqt5界面\processLicenceForbs\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1149, 653)
        font = QtGui.QFont()
        font.setPointSize(11)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("favicon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setAutoFillBackground(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.originPicLabel = QtWidgets.QLabel(self.centralwidget)
        self.originPicLabel.setGeometry(QtCore.QRect(40, 140, 500, 300))
        self.originPicLabel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.originPicLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.originPicLabel.setLineWidth(5)
        self.originPicLabel.setMidLineWidth(0)
        self.originPicLabel.setText("")
        self.originPicLabel.setScaledContents(True)
        self.originPicLabel.setObjectName("originPicLabel")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(240, 90, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Candara")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("background-color:rgb(170, 255, 127)")
        self.label_2.setObjectName("label_2")
        self.selectPicButton = QtWidgets.QPushButton(self.centralwidget)
        self.selectPicButton.setGeometry(QtCore.QRect(260, 500, 91, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.selectPicButton.setFont(font)
        self.selectPicButton.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.selectPicButton.setStyleSheet("background-color:rgb(170, 255, 127)")
        self.selectPicButton.setAutoDefault(True)
        self.selectPicButton.setDefault(False)
        self.selectPicButton.setObjectName("selectPicButton")
        self.recognizeButton = QtWidgets.QPushButton(self.centralwidget)
        self.recognizeButton.setGeometry(QtCore.QRect(420, 500, 91, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.recognizeButton.setFont(font)
        self.recognizeButton.setStyleSheet("background-color:rgb(170, 255, 127)")
        self.recognizeButton.setAutoDefault(True)
        self.recognizeButton.setObjectName("recognizeButton")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(780, 90, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Candara")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("background-color:rgb(170, 255, 127)")
        self.label_3.setObjectName("label_3")
        self.outputTextEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.outputTextEdit.setGeometry(QtCore.QRect(680, 130, 341, 321))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.outputTextEdit.setFont(font)
        self.outputTextEdit.setObjectName("outputTextEdit")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(50, 480, 141, 91))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setStyleSheet("background-color:rgb(170, 255, 127)")
        self.groupBox.setObjectName("groupBox")
        self.MLPCheck = QtWidgets.QRadioButton(self.groupBox)
        self.MLPCheck.setGeometry(QtCore.QRect(10, 30, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.MLPCheck.setFont(font)
        self.MLPCheck.setChecked(True)
        self.MLPCheck.setObjectName("MLPCheck")
        self.SVMCheck = QtWidgets.QRadioButton(self.groupBox)
        self.SVMCheck.setGeometry(QtCore.QRect(10, 60, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.SVMCheck.setFont(font)
        self.SVMCheck.setObjectName("SVMCheck")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(320, 10, 531, 61))
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("background-color:rgb(0, 170, 127)\n"
"")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.timeLabel = QtWidgets.QLabel(self.centralwidget)
        self.timeLabel.setGeometry(QtCore.QRect(40, 20, 211, 51))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.timeLabel.setFont(font)
        self.timeLabel.setAutoFillBackground(False)
        self.timeLabel.setStyleSheet("background-color:rgb(170, 255, 127)")
        self.timeLabel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.timeLabel.setScaledContents(False)
        self.timeLabel.setObjectName("timeLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1149, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.selectPicButton.clicked.connect(MainWindow.choosePicButtonClicked)
        self.recognizeButton.clicked.connect(MainWindow.recognizeButtonClicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "车牌识别系统"))
        self.label_2.setText(_translate("MainWindow", "识别对象"))
        self.selectPicButton.setText(_translate("MainWindow", "选择图片"))
        self.recognizeButton.setText(_translate("MainWindow", "识别"))
        self.label_3.setText(_translate("MainWindow", "处理过程"))
        self.outputTextEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:12pt; font-weight:600; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt; font-weight:400;\"><br /></p></body></html>"))
        self.groupBox.setTitle(_translate("MainWindow", "分类器选择："))
        self.MLPCheck.setText(_translate("MainWindow", "MLP分类器"))
        self.SVMCheck.setText(_translate("MainWindow", "SVM分类器"))
        self.label.setText(_translate("MainWindow", "车牌识别系统演示版"))
        self.timeLabel.setText(_translate("MainWindow", "NOW"))
