
from  Ui_mainwindow import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication,QMessageBox,QFileDialog
from PyQt5.QtGui import QImage, QPixmap,QPalette, QBrush
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot,QTimer
import datetime
import sys
import numpy as np
import cv2
import os
import matplotlib.colors
import matplotlib
from matplotlib import pyplot as plt
import math
from time import time 
from numpy import fft
from sklearn.decomposition import PCA
from functools import reduce
from matplotlib.pyplot import MultipleLocator
from sklearn import svm
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import profunc as prf
import net 
# import net_clean


class mywindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()#能调用父类的方法
        self.originimg = np.array([])#原始图片
        self.setupUi(self)
        #设置背景图片
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("backpic1.jpg")))
        self.setPalette(palette)
        #初始化定时器   
        self.timer()
        #显示
        self.show()
        #加载模型，初始化变量
        self.init()
     
    def init(self):
        #========================================================
        matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        #=======================初始化=============================
        self.picChooseFlag = 0
        self.outputTextEdit.append('加载SVM模型......\n')
        # QApplication.processEvents()
        # print('加载SVM模型......\n')
        self.svm_model = joblib.load('svm_model')
        self.outputTextEdit.append('加载神经网络模型......\n')
        # QApplication.processEvents()
        # print('加载神经网络......\n')
        # Hw = joblib.load('Hw')
        # Ow = joblib.load('Ow')
        self.Hw = joblib.load('Hw_0.3_20000')
        self.Ow = joblib.load('Ow_0.3_20000')

        # Hw = joblib.load('G:/我的地盘/毕设用/AA毕设\'s/神经网络/mes/Hw_0.3_20000')
        # Ow = joblib.load('G:/我的地盘/毕设用/AA毕设\'s/神经网络/mes/Ow_0.3_20000')
        # Hw = joblib.load('G:/我的地盘/毕设用/AA毕设\'s/神经网络/w_jcs_Pra_[20, 12]_Acc1.000_0.989_sigmoid')
        self.lossmode = 'j'#j:jcs m:mes
        self.outputTextEdit.append('加载PCA模型......\n')
        # QApplication.processEvents()
        # print('加载pca模型......\n')
        self.pca = joblib.load('pca_model')
        self.outputTextEdit.append('初始化完成\n')
        self.outputTextEdit.append('==================================')
        QApplication.processEvents()
        # print('加载完成\n')
        self.clas=np.array(['A','B','0','1','2','3','4','5','6','7','8','9']) 
        self.selectPicButton.setFocus()
    
    def recognizeProcess(self):
        self.outputTextEdit.append('==================================')
        QApplication.processEvents()
        realnum = (os.path.split(self.picadd)[1]).split('.')[0] #真实的车牌号
        #======================图像恢复===============================
        imgGray = cv2.cvtColor(self.originimg,cv2.COLOR_BGR2GRAY) 
        t = time()
        dis = prf.get_pra(imgGray)#模糊程度
        self.outputTextEdit.append('获取模糊参数：%.6f Seconds' % (time()-t))
        QApplication.processEvents()
        t = time()
        # print('模糊程度：\t ',dis)
        psf = prf.get_motion_dsf(imgGray.shape,0,int(dis))#获取点扩散函数
        self.outputTextEdit.append('产生PSF函数：%.6f Seconds' % (time()-t))
        QApplication.processEvents()
        t = time()
        imgRe = prf.wiener1(self.originimg,psf,channel=3) 
        self.outputTextEdit.append('维纳滤波用时：%.6f Seconds' % (time()-t))
        QApplication.processEvents()
        t = time()
        #=====================空间转换================================
        imgHSV =cv2.cvtColor(imgRe,cv2.COLOR_BGR2HSV)
        # prf.cvshow(u'recovered image',imgRe)
        vch = imgHSV[:,:,2]#提取V通道
        self.outputTextEdit.append('空间转换时间：%.6f Seconds' % (time()-t))
        QApplication.processEvents()
        # print('空间转换时间：%f Seconds' % (time()-t))
        t = time()
        #=====================车牌区域提取========================================
        ret,erzhi = cv2.threshold(vch,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # prf.cvshow(u'binerary image',erzhi)
        kernel = np.ones((5,5),np.uint8) 
        closing = cv2.morphologyEx(erzhi, cv2.MORPH_CLOSE, kernel)
        # prf.cvshow(u'closing ',closing)
        binary,cnts, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #轮廓选择
        # drawimg = imgRe.copy()
        # res = cv2.drawContours(drawimg, cnts, -1, (0, 0, 255), 2)
        # prf.cvshow('res',res)
        realcnt = np.array([])
        for i,x in enumerate(cnts):
            area = cv2.contourArea(x)
            if area>15000 and area<30000:
                realcnt = x
                break
        (x,y,w,h) = cv2.boundingRect(realcnt)
        chepai = imgRe[y:y+h,x:x+w,:]
        self.outputTextEdit.append('车牌定位时间：%.6f Seconds' % (time()-t))
        QApplication.processEvents()
        # print('车牌定位时间：%f Seconds' % (time()-t))
        t = time()
        #=============================车牌字符分割==================================
        # prf.cvshow(u'roi region',chepai)
        cpGray = cv2.cvtColor(chepai,cv2.COLOR_BGR2GRAY)
        mecp = cv2.medianBlur(cpGray,5)
        ret,bmcp = cv2.threshold(mecp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((30,12),np.uint8) 
        cbmcp = cv2.morphologyEx(bmcp, cv2.MORPH_CLOSE, kernel)
        # prf.cvshow(u'binerary region',cbmcp)
        binary,numcnts, hierarchy = cv2.findContours(cbmcp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #筛选下再 防止有噪音点
        numcandi = []
        for i,x in enumerate(numcnts):
            area = cv2.contourArea(x)
            if area>150 :
                numcandi.append(x) 
        numcandi = prf.sort_contours(numcandi, method="left-to-right")[0] #排序，从左到右，从上到下
        self.outputTextEdit.append('字符分割时间：%.6f Seconds' % (time()-t))
        QApplication.processEvents()
        # print('字符分割时间：%f Seconds' % (time()-t))
        t = time()
        #============================识别===========================================
        fig = plt.figure(figsize =(15,10) ,facecolor='w')
        fig.canvas.set_window_title('识别详情')
        result = []
        recogTime = 0
        for i,c in enumerate(numcandi):
            ts = time()
            (x, y, w, h) = cv2.boundingRect(c)	 #定位出数字坐标
            roi = mecp[y-2:y+h,x-2:x+w]          #抠出来数字
            roi = cv2.resize(roi,(30,30),interpolation=cv2.INTER_LINEAR)  
            x1 = roi.reshape(1,-1)
            if self.SVMCheck.isChecked:
                y =  self.svm_model.predict(self.pca.transform(x1))#主成分分析后的向量送进去预测使用SVM
            elif self.MLPCheck.isChecked:
                if self.lossmode == 'j':
                    y = net.netjcs_predict(self.pca.transform(x1),self.Hw,self.Ow)#使用损失函数是交叉熵的神经网络
                else:
                    y = net.net_predict(self.pca.transform(x1),self.Hw,self.Ow)#使用神经网络
            recogTime += (time()-ts)
            # y = net.net_pro_jiaocha_4_elu_predict(pca.transform(x1),Hw)
            # y = net_clean.net_pro_jiaocha_predict(pca.transform(x1),Hw,func= 'sigmoid',loss = 'jcs')
            result.append(self.clas[y][0])
            #y =  svm_model.predict(x1)
            plt.subplot(3,3,i+4)
            plt.title(u'分为：%s' % self.clas[y][0])
            plt.imshow(roi,cmap='gray') 

        result = "".join(result)
        output = ''
        for i,x in enumerate(result):
            if result[i] != realnum[i]:
                output += realnum[i]+'分为了' +result[i]+'\n'
                # self.wrongnum += 1
        plt.subplot(3,3,1)
        plt.title(u'原图' )
        # plt.imshow(imgGray,cmap='gray') 
        plt.imshow(self.pltshowtrans(self.originimg))      
        plt.subplot(3,3,2)
        plt.title(u'复原图' )
        plt.imshow(self.pltshowtrans(imgRe)) 
        plt.subplot(3,3,3)
        plt.title(u'车牌区域' )
        # plt.imshow(imgGray,cmap='gray') 
        plt.imshow(self.pltshowtrans(chepai))   
        # plt.imshow(imgRe,cmap='gray')

        plt.suptitle('识别结果:'+result+'\n'+'真实值:'+realnum+'\n'+output)  
        plt.tight_layout(1.7)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.8,wspace=None, hspace=0.5)
        self.outputTextEdit.append('识别分类时间：%.6f Seconds' % (recogTime))
        QApplication.processEvents()
        self.outputTextEdit.append('画图时间：%.6f Seconds' % (time()-t-recogTime))
        # print('识别画图时间：%f Seconds' % (time()-t))
        self.outputTextEdit.append('识别结果:'+result+'\n'+'真实值:'+realnum+'\n'+output)
        self.outputTextEdit.append('==================================')
        # QApplication.processEvents()
        plt.show()
        # print('识别结果:'+result+'\n'+'真实值:'+realnum+'\n'+output)
       

    def pltshowtrans(self,img):
        b,g,r = cv2.split(img)
        return cv2.merge((r,g,b))


    def choosePicButtonClicked(self):
        #F:\毕设用\AA毕设\模糊矿车图片
        self.picadd = QFileDialog.getOpenFileName(self,"选择图片","F:\毕设用\AA毕设\模糊矿车图片","图片|*.jpg;*.bmp;*.png")[0]
        if(len(self.picadd)):
            self.originimg = cv2.imdecode(np.fromfile(self.picadd,dtype=np.uint8),-1)#读取
            self.originPicLabel.setPixmap(QPixmap.fromImage(self.showTrans(self.originimg)))
            self.picChooseFlag = 1
            self.recognizeButton.setFocus()
        else:
            self.selectPicButton.setFocus()

    
    def recognizeButtonClicked(self):
        if self.picChooseFlag == 1:  
            # self.t = threading.Thread(target=self.recognizeProcess, name='funciton')
            # self.t.start()
            self.recognizeProcess()
            self.picChooseFlag = 0
        elif self.picChooseFlag ==0:
            QMessageBox.about(self, '提示','请先选择要识别的图片哦')
        self.selectPicButton.setFocus()

    def showTrans(self,img):
        # 提取图像的通道和尺寸，用于将OpenCV下的image转换成Qimage
        height, width, channel = img.shape
        bytesPerline = 3 * width
        return  QImage(img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()

    def closeEvent(self,event):
        reply = QMessageBox.question(self, '退出', '确认退出吗', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
      
    def timer(self):
        timer = QTimer(self)
        timer.timeout.connect(self.refreshTime)
        timer.start(300)
   
    def refreshTime(self):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.timeLabel.setText(now)

# class myThread (threading.Thread):
#     def __init__(self, threadID, name):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name

        
if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    mine = mywindow()
    sys.exit(app.exec_())