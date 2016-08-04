# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:46:17 2016

@author: crw
"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage 
import sys
import time
import shutil
font = cv2.FONT_HERSHEY_SIMPLEX
import caffe
from skimage import transform as tf

caffe.set_mode_cpu()
# 训练数据中，每个通道的平均值
averageImg = [103.939, 116.779, 128.68]

# 全局使用到的一些数据，保留在全局变量
#=====================================
model_path_model ='./'
model_path_deploy ='./'
model_define=model_path_deploy+'deploy.prototxt'
model_weight=model_path_model+'maxout128x128_iter_210000.caffemodel'
image_formats =['jpg','png','bmp']

data_w = 128
data_h =  128
scale = 255
data_as_grey = False
sub_mean = False

predict_layer1 ='gender_prob'
net = caffe.Classifier(model_define, model_weight)
#====================================
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
    if image.ndim == 2:
        image_au = np.empty((w,h,3))
        image_au[:,:,0]=image
        image_au[:,:,1]=image
        image_au[:,:,2]=image
        image = image_au
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

def get_predict(filename):
    '''
    @brief：获取特征
    @param： 图像的文件
    @return：feature，提取到的人脸特征
    '''
    X=read_image(filename,w=data_w,h=data_h,as_grey=data_as_grey)    
    out = net.forward_all(data=X)                                                      
    predict_1 = np.float64(out[predict_layer1])
    return predict_1

def evaluate_list(file_list, image_path, result, image_path_affi):
    fid = open(file_list)
    fid_result = open(result,"w")
    fid.readline()
    for line in fid:
        items = line.split(',')
        file_name = items[0]
        print "processing ",file_name
        x = items[1]
        y = items[2]
        w = items[3]
        h = items[4]
        true_file_path =""
        if not os.path.isfile(image_path+file_name):
            true_file_path = image_path_affi+file_name
        else:
            true_file_path = image_path+file_name
        predict_1, predict_2 = get_predict(true_file_path)
        
        gender = "1" if predict_1[0][1]>0.5 else "0"
        smile = "1" if predict_2[0][1]>0.5 else "0"
        fid_result.write(file_name+","+x+","+y+","+w+","+h+","+gender+","+smile+"\n")
    fid.close()
    fid_result.close()

def evaluate_path(file_path, save_path_gendr):
    file_lists = os.listdir(file_path)
    for file_ in file_lists:
        file_name = os.path.join(file_path,file_)
        predict_1 = get_predict(file_name)
        gender = "1" if predict_1[0][1]>0.5 else "0"
        print file_,"gender:",gender
        shutil.copy2(file_name,os.path.join(save_path_gendr,gender,file_))
        
if __name__ == '__main__':

    file_path = "E:/everphoto/test_online_image_face_0.5/"
    save_path_gendr = "./result/"

    evaluate_path(file_path,save_path_gendr)
    
