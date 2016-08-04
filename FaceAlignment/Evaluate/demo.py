# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 20:39:22 2014

@author: crw
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

caffe_root = '/home/crw/caffe-local/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2
# 指定一些图形显示的参数;
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# 定义显示函数;
def vis_square(data, padsize=1, padval=0):
    # 归一化到0-1
    data -= data.min()
    data /= data.max()
    
    # 强制使得滤波器为方形;
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # 把滤波器序列转换成图像显示;
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    #显示数据：
    plt.imshow(data)

    
itera=25000
net = caffe.Classifier(caffe_root+'faceexp/try1_3/architecture.prototxt',
                                   '/media/crw/DataCenter/ResultModel/faceexp/try1_3/snapshot_iter_' +str(itera)+'.caffemodel')

#指定网络的一个格式;
net.set_phase_test()
net.set_mode_cpu()

j=1
result=np.zeros((64,64,64))
testImageNum=32
for i in range(testImageNum):
    image=caffe.io.load_image('/media/crw/DataCenter/TestImage/FaceExp/' +str(i+1)+'.jpg')
    img=np.reshape(image,(64,64,3))
    image=image*255

    scores = net.predict([image], False)
    result[j-1,:,:]=img*255
    
    j=j+1
    newscores= (scores[:,:]-scores.min())/scores.max()*255
    reconstructpic=np.reshape(newscores,(64,64))
    
    result[j-1,:,:]=reconstructpic
    j=j+1

plt.figure()
plt.axis('off')
vis_square(result,padsize=5, padval=0)
plt.xlabel('Figure 1. The First Column is the origion image, the Second Column is the reconstruct image')
plt.savefig('result'+str(itera)+'.jpg')
