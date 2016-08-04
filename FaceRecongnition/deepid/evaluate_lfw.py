# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:55:55 2015

@author: 陈日伟 <riwei.chen@outlook.com>
@brief：在lfw数据库上验证训练好了的网络
"""
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import skimage 
import sys
caffe_root = '/home/crw/caffe-master/' 
sys.path.insert(0, caffe_root + 'python')
import caffe
import sklearn.metrics.pairwise as pw

def read_imagelist(filelist):
    '''
    @brief：从列表文件中，读取图像数据到矩阵文件中
    @param：filelist 图像列表文件
    @return：4D 的矩阵
    '''
    fid=open(filelist)
    lines=fid.readlines()
    test_num=len(lines)
    fid.close()
    X=np.empty((test_num,3,64,64))
    i =0
    for line in lines:
        word=line.split('\n')
        filename=word[0]
        im1=skimage.io.imread(filename,as_grey=False)
        image =skimage.transform.resize(im1,(64, 64))*255
        if image.ndim<3:
            print 'gray:'+filename
            X[i,0,:,:]=image[:,:]
            X[i,1,:,:]=image[:,:]
            X[i,2,:,:]=image[:,:]
        else:
            X[i,0,:,:]=image[:,:,0]
            X[i,1,:,:]=image[:,:,1]
            X[i,2,:,:]=image[:,:,2]
        i=i+1
    return X

def read_labels(labelfile):
    '''
    读取标签列表文件
    '''
    fin=open(labelfile)
    lines=fin.readlines()
    labels=np.empty((len(lines),))
    k=0;
    for line in lines:
        labels[k]=int(line)
        k=k+1;
    fin.close()
    return labels

def draw_roc_curve(fpr,tpr,title='cosine',save_name='roc_lfw'):
    '''
    画ROC曲线图
    '''
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic using: '+title)
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(save_name+'.png')


def evaluate(itera=500000,metric='cosine'):
    '''
    @brief: 评测模型的性能
    @param：itera： 模型的迭代次数
    @param：metric： 度量的方法
    '''
     # 转换均值图像数据　-->npy格式文件
    fin='/media/crw/MyBook/TrainData/LMDB/CASIA-WebFace/10575_64X64/mean.binaryproto'
    fout='/media/crw/MyBook/TrainData/LMDB/CASIA-WebFace/10575_64X64/mean.npy'
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( fin , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]
    np.save( fout , out )
    #设置为gpu格式
    caffe.set_mode_gpu()
    net = caffe.Classifier('./deploy.prototxt', 
    '/media/crw/MyBook/Model/FaceRecognition/try5_2/snapshot_iter_'+str(itera)+'.caffemodel',
    mean=np.load(fout))
    #需要对比的图像，一一对应
    filelist_left='./LFW_Test_List/left.list'
    filelist_right='./LFW_Test_List/right.list'
    filelist_label='./LFW_Test_List/label.list'
    
    print 'network input :' ,net.inputs  
    print 'network output： ', net.outputs
    #提取左半部分的特征
    X=read_imagelist(filelist_left)
    test_num=np.shape(X)[0]
    #data_1 是输入层的名字
    out = net.forward_all(data_1 = X)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    feature1 = np.float64(out['deepid_1'])
    feature1=np.reshape(feature1,(test_num,160))
    #np.savetxt('feature1.txt', feature1, delimiter=',')

    #提取右半部分的特征
    X=read_imagelist(filelist_right)
    out = net.forward_all(data_1=X)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    feature2 = np.float64(out['deepid_1'])
    feature2=np.reshape(feature2,(test_num,160))
    #np.savetxt('feature2.txt', feature2, delimiter=',')
    
    #提取标签    
    labels=read_labels(filelist_label)
    assert(len(labels)==test_num)
    #计算每个特征之间的距离
    mt=pw.pairwise_distances(feature1, feature2, metric=metric)
    predicts=np.empty((test_num,))
    for i in range(test_num):
          predicts[i]=mt[i][i]
        # 距离需要归一化到0--1,与标签0-1匹配
    for i in range(test_num):
            predicts[i]=(predicts[i]-np.min(predicts))/(np.max(predicts)-np.min(predicts))
    print 'accuracy is :',calculate_accuracy(predicts,labels,test_num)
            	 
    np.savetxt('predict.txt',predicts)        	 
    fpr, tpr, thresholds=sklearn.metrics.roc_curve(labels,predicts)
    draw_roc_curve(fpr,tpr,title=metric,save_name='lfw_'+str(itera))
    
def calculate_accuracy(distance,labels,num):    
    '''
    #计算识别率,
    选取阈值，计算识别率
    '''    
    accuracy = []
    predict = np.empty((num,))
    threshold = 0.2
    while threshold <= 0.8 :
        for i in range(num):
            if distance[i] >= threshold:
            	 predict[i] =1
            else:
            	 predict[i] =0
        predict_right =0.0
        for i in range(num):
        	if predict[i]==labels[i]:
        	  predict_right = 1.0+predict_right
        current_accuracy = (predict_right/num)
        accuracy.append(current_accuracy)
        threshold=threshold+0.001
    return np.max(accuracy)

if __name__=='__main__':
    itera=500000
    metric='cosine'
    evaluate(itera,metric)
