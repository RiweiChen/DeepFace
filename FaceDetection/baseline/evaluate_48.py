# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from  math import pow
from skimage import transform as tf
import cv2
import caffe
from nms import nms_average,nms_max
import render_result

caffe.set_mode_cpu()
#caffe.set_device(0)

model_path = './'
model_define= model_path+'deploy.prototxt'
model_weight =model_path+'snapshot_iter_100000.caffemodel'
model_define_fc =model_path+'deploy_fc.prototxt'
model_weight_fc =model_path+'snapshot_iter_100000_fc.caffemodel'

channel = 3
raw_scale = 255.0
face_w = 48
stride = 16
cellSize = face_w
threshold = 0.95
factor = 0.793700526 # 缩小因子
MAX_W = 1000
map_idx = 0
params = ['deepid', 'fc7']
params_fc =  ['deepid-conv', 'fc7-conv']


def generateBoundingBox(featureMap, scale):
    boundingBox = []
    for (y,x), prob in np.ndenumerate(featureMap):
       if(prob >= threshold):
            x-=1
            y-=1
            # format: (x1,y1,x2,y2)
            boundingBox.append([float(stride * x)/scale, float(stride *y )/scale, 
                              float(stride * x + cellSize - 1)/scale, float(stride * y + cellSize - 1)/scale, prob])
    return boundingBox

def convert_full_conv(model_define,model_weight,model_define_fc,model_weight_fc):
    '''
    @breif: 将原始网络转换为全卷积模型
    @param: model_define,二分类网络定义文件
    @param: model_weight，二分类网络训练好的参数
    @param: model_define_fc,生成的全卷积网络定义文件
    @param: model_weight_fc，转化好的全卷积网络的参数
    '''
    net = caffe.Net(model_define, model_weight, caffe.TEST)
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
    net_fc = caffe.Net(model_define_fc, model_weight, caffe.TEST)
    conv_params = {pr: (net_fc.params[pr][0].data, net_fc.params[pr][1].data) for pr in params_fc}
    for pr, pr_conv in zip(params, params_fc):
       conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
       conv_params[pr_conv][1][...] = fc_params[pr][1]
    net_fc.save(model_weight_fc)
    print 'convert done!'
    return net_fc

def show_heatmap(heatmaps):
    num_scale = len(heatmaps)
    s=int(np.sqrt(num_scale))+1
    for idx,heatmap in enumerate(heatmaps):
        plt.subplot(s, s+1, idx+1)
        plt.axis('off')
        plt.imshow(heatmap)
    #plt.savefig('heatmap/'+image_name.split('/')[-1])

def caculate_scales(imgs):
    scales = []
    if imgs.ndim==3:
        rows,cols,ch = imgs.shape
    else:
        rows,cols = imgs.shape
    min_ = rows if  rows<=cols  else  cols
    max_ = rows if  rows>=cols  else  cols
    # 放大的尺度    
    delim = MAX_W/max_
    while (delim >= 1):
        scales.append(delim)
        delim=delim-0.5
    # 缩小的尺度
    min_ = min_ * factor
    factor_count = 1
    while(min_ >= face_w):
        scale = pow(factor,  factor_count)
        scales.append(scale)
        min_ = min_ * factor
        factor_count += 1
    scales.append(1)
    return scales 

def face_detection_image(net,image_name):
    imgs = cv2.imread(image_name)
    rows,cols,ch = imgs.shape
    scales = caculate_scales(imgs)
    total_boxes = []
    for scale in scales:
        w,h = int(rows* scale),int(cols* scale)
        scale_img = tf.resize(imgs,(w,h))
        #scale_img = cv2.resize(imgs,(w,h))/255.0
        net.blobs['data'].reshape(1,channel,w,h)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', raw_scale)
        out = net.forward_all(data=np.asarray([transformer.preprocess('data', scale_img)]))
        boxes = generateBoundingBox(out['prob'][0,map_idx], scale)
        if(boxes):
            total_boxes.extend(boxes)
    boxes_nms = np.array(total_boxes)
    true_boxes = nms_max(boxes_nms, overlapThresh=0.3)
    true_boxes = nms_average(np.array(true_boxes), overlapThresh=0.07)
    img_cv = render_result.read_image(image_name)
    result = img_cv
    for box in true_boxes:
        result = render_result.draw_rectangle(result,(int(box[0]),int(box[1]),int(box[2]),int(box[3])),(0,0,255))
    render_result.save_2_file(result,'result/'+image_name.split('/')[-1])
    return true_boxes
    
if __name__ == "__main__":
    if not os.path.isfile(model_weight_fc):
        net_fc = convert_full_conv(model_define,model_weight,model_define_fc,model_weight_fc)
    else:
        net_fc = caffe.Net(model_define_fc, model_weight_fc, caffe.TEST) 
    filename = "./images/2.jpg"
    fms = face_detection_image(net_fc,filename)
    for fm in fms:
        print(fm)
    plt.close('all')
