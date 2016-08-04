# -*- coding: utf-8 -*-
import os
import time
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pylab as plt

from deepface import face_verify
from deepface import face_smile

from sklearn import manifold
from sklearn import svm

def demo_verification(image1,image2):
    image1='testimage/8.jpg'
    image2='testimage/ozwtest2.jpg'
    start = time.time()
    f1 = face_verify.get_image_feature(image1)
    f2 = face_verify.get_image_feature(image2)
    print f1
    result = face_verify.evaluate_by_distance(f1, f2)
    sim = face_verify.get_similarity(result, face_verify.threshold)
    if sim <= 0.5 :
        print 'predict: same person'
    else:
        print 'predict: different person'
    cost_time = time.time()-start
    print 'It take:  %f second to complete'%(cost_time)

def get_face_features(file_path):
    file_list =os.listdir(file_path)
    features =dict()
    for file_ in file_list:
        print file_
        file_name = os.path.join(file_path, file_)
        feature = face_verify.get_image_feature(file_name)
        feature = np.reshape(feature,(256,))
        features[file_] =feature
    return features

def clusting_face(features):
    est = KMeans(n_clusters=2)
    est.fit(features)
    labels = est.labels_
    return labels

def outier_detect_one_svm(features):
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(features)
    labels = clf.predict(features)
    return labels

def feature_Visualizing(features,labels):
    num = len(labels)
    Y = np.zeros((num,))
    '''
    for i,label in enumerate(labels):
        if label == -1:
            Y[i] = 0
        else:
            Y[i] = 1
    '''
    F = np.zeros((num,np.shape(features)[1]))
    for i in range(num):
        F[i,:] = features[i]
        Y[i] = labels[i]
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X = tsne.fit_transform(F)
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的labels
    words ='01'
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1],words[int(Y[i])],
                 color=plt.cm.Set1(Y[i]/(num*1.0)),
                 fontdict={'weight': 'bold', 'size': 9})
    #plt.axis('off')
    plt.savefig('result.jpg')
    plt.show()

def test_smile():
    file_path = "D:/MyDataset/dlib_aligned/Genki4K"
    file_list = os.listdir(file_path)
    for file_ in file_list:
        full_file = os.path.join(file_path, file_)
        predict = face_smile.predict(full_file)
        print "smile predict: ",predict
    file_name  = "testimage/file2867.jpg"
    predict = face_smile.predict(file_name)
    print "smile predict: ",predict	
if __name__ == '__main__':
    test_smile()
    

	
	
