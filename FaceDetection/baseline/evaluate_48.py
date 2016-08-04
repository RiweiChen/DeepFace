# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
from  math import pow
import skimage.io
from skimage import transform as tf

caffe_root = '/media/crw/MyBook/Caffe/caffe-triplet/'
sys.path.insert(0, caffe_root + 'python')
import caffe

from nms import nms_average,nms_max

caffe.set_device(0)
caffe.set_mode_gpu()

model_path = '/media/crw/MyBook/FaceModel/FaceDetection/try1_4/'
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

map_idx = 0
params = ['deepid', 'fc7']
params_fc =  ['deepid-conv', 'fc7-conv']

def generateBoundingBox(featureMap, scale):
    '''
    @brief: 生成窗口
    @param: featureMap,特征图，scale：尺度
    '''
    boundingBox = []
    for (x,y), prob in np.ndenumerate(featureMap):
       if(prob >= threshold):
           # 映射到原始的图像中的大小
            x=x-1
            y=y-1
            boundingBox.append([float(stride * y)/scale, float(stride *x )/scale, 
                              float(stride * y + cellSize - 1)/scale, float(stride * x + cellSize - 1)/scale, prob])
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

def re_verify(net_vf, img):
    '''
    @breif: 对检测到的目标框进行重新的验证
    '''
    img= tf.resize(img,(face_w,face_w))
    transformer = caffe.io.Transformer({'data': net_vf.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', raw_scale)
    out = net_vf.forward_all(data=np.asarray([transformer.preprocess('data', img)]))
    #print out['prob']
    if out['prob'][0,map_idx] > threshold:
        return True
    else:
        return False
        
def face_detection_image(net,net_vf,image_name):
    '''
    @brief: 检测单张人脸图像
    '''
    scales = []
    imgs = skimage.io.imread(image_name)
    if imgs.ndim==3:
            rows,cols,ch = imgs.shape
    else:
            rows,cols = imgs.shape
    # 计算需要的检测的尺度因子
    min = rows if  rows<=cols  else  cols
    max = rows if  rows>=cols  else  cols
    # 放大的尺度    
    delim = 2500/max
    while (delim >= 1):
        scales.append(delim)
        delim=delim-0.5
    # 缩小的尺度
    min = min * factor
    factor_count = 1
    while(min >= face_w):
        scale = pow(factor,  factor_count)
        scales.append(scale)
        min = min * factor
        factor_count += 1
    #=========================
    #scales.append(1)
    total_boxes = []
    ###显示热图用
    num_scale = len(scales)
    s1=int(np.sqrt(num_scale))+1
    tt=1
    plt.subplot(s1, s1+1, tt)
    plt.axis('off')
    plt.title("Input Image")
    im=caffe.io.load_image(image_name)
    plt.imshow(im)
    #============
    for scale in scales:
        w,h = int(rows* scale),int(cols* scale)
        scale_img= tf.resize(imgs,(w,h))
        # 更改网络输入data图像的大小
        net.blobs['data'].reshape(1,channel,w,h)
        # 转换结构
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        #transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', raw_scale)
        out = net.forward_all(data=np.asarray([transformer.preprocess('data', scale_img)]))
        ###显示热图用
        tt=tt+1
        plt.subplot(s1, s1+1, tt)
        plt.axis('off')
        plt.title("sacle: "+ "%.2f" %scale)
        plt.imshow(out['prob'][0,map_idx])
        #===========
        boxes = generateBoundingBox(out['prob'][0,map_idx], scale)
        if(boxes):
            total_boxes.extend(boxes)
    # 非极大值抑制
    boxes_nms = np.array(total_boxes)
    true_boxes1 = nms_max(boxes_nms, overlapThresh=0.3)
    true_boxes = nms_average(np.array(true_boxes1), overlapThresh=0.07)
    #===================
    plt.savefig('heatmap/'+image_name.split('/')[-1])
    # 在图像中画出检测到的人脸框
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(imgs)
    for box in true_boxes:
        im_crop = im[box[0]:box[2],box[1]:box[3],:]
        if im_crop.shape[0] == 0 or im_crop.shape[1] == 0:
            continue
        if re_verify(net_vf, im_crop) == True:
            rect = mpatches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                fill=False, edgecolor='red', linewidth=1)
            ax.text(box[0], box[1]+20,"{0:.3f}".format(box[4]),color='white', fontsize=6)
            ax.add_patch(rect)
    plt.savefig('result/'+image_name.split('/')[-1])
    plt.close()
    return out['prob'][0,map_idx]
    
    
if __name__ == "__main__":
    if not os.path.isfile(model_weight_fc):
        net_fc = convert_full_conv(model_define,model_weight,model_define_fc,model_weight_fc)
    else:
        net_fc = caffe.Net(model_define_fc, model_weight_fc, caffe.TEST)
    net_vf = caffe.Net(model_define, model_weight, caffe.TEST)
    for i in range(210):
        image_name = '/media/crw/MyBook/TestData/myDataBase/'+str(i+1)+'.jpg'
        print i
        fm = face_detection_image(net_fc,net_vf,image_name)
        plt.close('all')
