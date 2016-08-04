# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 20:39:22 2014

@author: crw
"""
import PIL.Image as Image
import numpy as np
import os.path
import sys
caffe_root = '/home/crw/caffe-local/'
sys.path.insert(0, caffe_root + 'python')
import caffe
net = caffe.Classifier(caffe_root+'faceexp/try1_2/face_exp.prototxt',
                                   '/media/crw/DataCenter/ResultModel/faceexp/try1_2/snapshot_iter_30000.caffemodel')
net.set_phase_test()
net.set_mode_cpu()

j=1
BASE_PATH='/media/crw/DataCenter/TestImage/FRGC100'
for dirname, dirnames, filenames in os.walk(BASE_PATH):
    for subdirname in dirnames: # 每一类。
        subject_path = os.path.join(dirname, subdirname)
        os.mkdir('/media/crw/DataCenter/ResultImage/faceexp/try1_2/FRGC/'+subdirname)
        j=1
        for filename in os.listdir(subject_path):#每一个图片
            abs_path = "%s/%s" % (subject_path, filename)
            image=caffe.io.load_image(abs_path)
            image=image*255
            scores = net.predict([image], False)
            newscores= (scores[:,:]-scores.min())/scores.max()*255
            reconstructpic=np.reshape(newscores,(64,64))
            img=Image.fromarray(reconstructpic)
            img = img.convert('RGB')
            
            savename='/media/crw/DataCenter/ResultImage/faceexp/try1_2/FRGC/'+subdirname+'/%d.jpg' %( j )
            j=j+1;
            img.save(savename)
            print abs_path
                
print 'done!'