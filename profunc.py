import numpy as np
import cv2
import os
import matplotlib.colors
import matplotlib.pyplot as plt
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

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.bmp','.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
def cvshow(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def get_motion_dsf(image_size, motion_angle, motion_dis):
    PSF = np.zeros(image_size)  # 点扩散函数
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2
 
    sin_val = math.sin(motion_angle * math.pi / 180)
    cos_val = math.cos(motion_angle * math.pi / 180)
 
    # 将对应角度上motion_dis个点置成1
    for i in range(motion_dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        PSF[int(x_center - x_offset), int(y_center + y_offset)] = 1
    PSF = PSF / PSF.sum() 
    # return fft.fft2(PSF)    # 归一化
    return PSF

#获取PSF参数
def get_pra(input):

    #way2
    input_fft=fft.fft2(input)
    input_absfft = np.abs(input_fft)
    h = np.log10(1+input_absfft)
    re1 = np.float_power(h,2)
    re1 = (fft.ifft2(re1))
    re = np.abs(fft.fftshift(re1))
    # print (re)
    xc = int(re.shape[0]/2)
    yc = int(re.shape[1]/2)
    right = re.copy()
    right[:,0:yc+5] = 0
    x0,y0 = np.unravel_index(np.argmax(re),re.shape)
    x1,y1 = np.unravel_index(np.argmax(right),re.shape)
    # print(x0,y0)
    # print(x1,y1)
    #这个角度只能求-90到90度的
    alpha = math.atan((x0-x1)/(y1-y0))*180/math.pi
    #若角度是0度，能求出来模糊像素的长度,0度和180度的方向分不开
    le = math.sqrt(math.pow((x0-x1),2)+math.pow((y1-y0),2))
    # d = le/(math.sin(math.atan((x0-x1)/(y1-y0)))) 

    # print('角度'),print(alpha,'\n')
    # print('长度'),print(le)

    return le 

def make_blurred(input, PSF, eps ,channel = 1):
    if channel == 1:
        input_fft = fft.fft2(input)# 进行二维数组的傅里叶变换
        PSF_fft = fft.fft2(PSF)#+ eps
        blurred = fft.ifft2(input_fft * PSF_fft)
        blurred = np.abs(fft.fftshift(blurred))
        return blurred
    elif channel == 3:
        ch = cv2.split(input)
        blurred = np.ones_like(ch)
        for i,x in enumerate(ch):
            input_fft = fft.fft2(x)# 进行二维数组的傅里叶变换
            PSF_fft = fft.fft2(PSF)+eps
            blur = fft.ifft2(input_fft * PSF_fft)
            blurred[i] = np.abs(fft.fftshift(blur))
        return cv2.merge((blurred[0],blurred[1],blurred[2]))#RGB


def wiener1(input,PSF,eps=0.001,SNR=0.001,channel = 1):        #维纳滤波，SNR=0.01
    if channel == 1:
        input_fft=fft.fft2(input)
        PSF_fft=fft.fft2(PSF) #+eps
        PSF_fft_1=np.conj(PSF_fft) /(np.abs(PSF_fft)**2 + SNR)
        result=fft.ifft2(input_fft * PSF_fft_1)
        result=np.abs(fft.fftshift(result))
        return result  
    elif channel == 3:
        ch = cv2.split(input)
        recov = np.ones_like(ch)
        for i ,x in enumerate(ch):
            input_fft=fft.fft2(x)
            PSF_fft=fft.fft2(PSF) #+eps
            PSF_fft_1=np.conj(PSF_fft) /(np.abs(PSF_fft)**2 + SNR)
            result=fft.ifft2(input_fft * PSF_fft_1)
            recov[i]=np.abs(fft.fftshift(result))
        return cv2.merge((recov[0],recov[1],recov[2]))#BGR
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] #用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b: b[1][i], reverse=reverse))
    # (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    # key=lambda b: b[1][1], reverse=reverse))#先左右在低高

    return cnts, boundingBoxes
def sort_contours_complicated(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] #用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key= lambda b: (b[1][1],b[1][0]), reverse=reverse))#先从低到高，再从左到右

    return cnts, boundingBoxes

