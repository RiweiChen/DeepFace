# -*- coding: utf-8 -*-
'''
@brief: 人脸微笑检测
@author:  Riwei Chen <riwei.chen@outlook.com>

Use in windows
'''
import math
import numpy as np
import skimage 

import sklearn.metrics.pairwise as pw
from skimage import transform as tf

import caffe
from model_config_smile import *
def read_image(filename,w=128,h=128,as_grey=False):
    '''
    @brief: 读取一个图片，返回矩阵
    @param：w，h：保留的图像大小
    '''
    if as_grey == True:
        X=np.empty((1,1,w,h))
    else:
        X=np.empty((1,3,w,h))
    image=skimage.io.imread(filename,as_grey=as_grey)
    #  注意输入图像是 0--1 ，还是0--255 的区别
    image=tf.resize(image,(w,h))*scale
    if as_grey == True:
        X[0,0,:,:]=image[:,:]
    else:
        # 注意通道的一致性
        if sub_mean == True:
            X[0,2,:,:]=image[:,:,0]-averageImg[0]
            X[0,1,:,:]=image[:,:,1]-averageImg[1]
            X[0,0,:,:]=image[:,:,2]-averageImg[2] 		
        else:
            X[0,2,:,:]=image[:,:,0]
            X[0,1,:,:]=image[:,:,1]
            X[0,0,:,:]=image[:,:,2]
    return X

def predict(file_name):
    '''
    @brief：根据传入的图像，进行人脸微笑判断。。
    @param： image图像的文件,range(0,1)
    @return：人脸微笑及其预测值。
    '''
    X = read_image(file_name,data_w,data_h,data_as_gray)
    out = net.forward_all(data=X)                              
    p = np.float64(out[feature_layer])
    return p[0][1]