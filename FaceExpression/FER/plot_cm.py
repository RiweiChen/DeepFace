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
from sklearn.metrics import confusion_matrix
font = cv2.FONT_HERSHEY_SIMPLEX
# caffe_root = '/media/crw/MyBook/Caffe/caffe-triplet/'
#sys.path.insert(0, caffe_root + 'python')
import caffe
from skimage import transform as tf

caffe.set_mode_cpu()
# 训练数据中，每个通道的平均值
averageImg = [103.939, 116.779, 128.68]

# 全局使用到的一些数据，保留在全局变量
#=====================================
#model_path_model ='/data01/chenriwei/Experiment/FaceExpression/FER/baseline/model/'
#model_path_deploy ='/data01/chenriwei/Experiment/FaceExpression/FER/baseline/'
model_path_model ='./'
model_path_deploy ='./'
model_define=model_path_deploy+'deploy.prototxt'
model_weight=model_path_model+'snapshot_iter_500000.caffemodel'

data_w = 42
data_h =  42
scale = 255
data_as_grey = True
sub_mean = False

predict_layer ='softmax'
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
    predict = np.float64(out[predict_layer])
    p = np.argmax(predict)
    return p

def plot_cm(cm, class_num):
    '''
    @brief： 打印出混淆矩阵
    '''
    print class_num
    print cm
    for idx,value in enumerate(class_num):
        if value == 0:
            continue
        print value
        for j in range(7):
            # ? what the problem ?
            print cm[idx][j]
            cm[idx][j] = (1.0*cm[idx][j])/value
            print cm[idx][j]
        
    # cm = 1.0*cm/test_total
    print cm
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(cm, cmap=plt.cm.jet,
                    interpolation='nearest')
    width = len(cm)
    height = len(cm[0])
    cb = fig.colorbar(res)
    ticks = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Predict Face Expression')
    plt.ylabel('Ground True Face Expression')
    plt.grid()
    locs, labels = plt.xticks(range(width), ticks)
    plt.yticks(range(height), ticks)
    plt.savefig('cm.eps', format='eps')

if __name__ == '__main__':
    test_file_list = 'E:/MyDataset/FaceExpressionSum/ICML2013Challenge/test.list'
    root_paht = "E:/MyDataset/FaceExpressionSum/ICML2013Challenge/Test/"
    gt = []
    pt = []
    test_num = 0
    class_num = [0,0,0,0,0,0,0]
    with open(test_file_list) as fid:
        for line in fid:
            items = line.split('\t')
            image_name = items[0]
            ground_true = int(items[1])
            predict = get_predict(os.path.join(root_paht,image_name))
            gt.append(ground_true)
            pt.append(predict)
            class_num[ground_true]+=1
            test_num+=1
            print test_num
    cm = confusion_matrix(pt, gt)
    plot_cm(cm, class_num)
