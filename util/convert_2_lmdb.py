# -*- coding: utf-8 -*-
'''
@brief 将训练数据转化为lmdb。
@author riwei.chen
@update
    2016.04.27: 添加no_index版本。
    2016.05.02: 去除no_index
'''
import os
TOOLS = '/home/chenriwei/Caffe/caffe-master-160503/build/tools/'	
#TOOLS ='/home/liuchundian/work/caffe3/.build_release/tools/'
#TOOLS = '/home/chenriwei/Caffe/caffe/.build_release/tools/'	
save_image_size = (128,128)
def div_train_val(file_in, file_train,file_val, val_rate = 10):
    '''
	@brief 将list训练数据划分为训练数据和验证数据集
	@param file_in: 输入的文件夹列表，
	@param file_train: 输出的训练列表
	@param file_val: 输出的验证列表
	@param val_rate: 测试数据的比例，即1/val_rate的数据为val数据
    '''
    i  = 0
    fid_train = open(file_train, 'w')
    fid_val = open(file_val,'w')
    for line in open(file_in):
        if i%val_rate == 0:
            fid_val.write(line)
        else:
            fid_train.write(line)
        i+=1
    fid_train.close()
    fid_val.close()
    
def convert_for_lmdb(input_file, output_file,Image_Root):
    '''
	@brief 将ImageData格式的list文件转化为LMDB所需要的格式文件，即是否为全路径问题。
	@param input_file: 输入的原始ImageData文件list列表
	@param output_file: 输出保存的位LMDB准备的文件列表
	
	@return None
    '''
    fid_in = open(input_file)
    fid_out = open(output_file, 'w')
    for line in fid_in.readlines():
        items = line.split('\t')
        file_path = items[0]
        file_path = file_path.split(Image_Root)[1]
        fid_out.write(file_path+'\t'+items[1])
    fid_in.close()
    fid_out.close()

def convert_2_lmdb(Image_Root, file_list_name, lmdb_path, save_image_size=(128,128)):
    '''
	@brief 将list文件列表转化为lmdb格式文件
	@param Image_Root: 图像的跟路径
	@param file_list_name: 图像列表的文件名
	@param imdb_path: 保存的imdb格式的文件路径
	@param save_image_size: 保存的图像大小
	'''
    command = 'GLOG_logtostderr=1 '+TOOLS+'convert_imageset --resize_height='+str(save_image_size[0])+' --resize_width='+str(save_image_size[1])  +'  --shuffle '+Image_Root+' ' +file_list_name+' '+lmdb_path
    print command
    os.system(command)
    print 'convert done!'

def caculate_image_mean(lmdb_path, save_path):
    command_mean =TOOLS+'compute_image_mean '+  lmdb_path+' '+save_path
    os.system(command_mean)
    print 'caculate image mean done!'

if __name__ == "__main__":	
    Image_Root = '/temp-hdd/chenriwei/Dataset/OnlineImage/margin_0.5_256/'
    
    lmdb_path = '/home/chenriwei/LMDB/OnlineImage/gender/margin_0.5_128_top3/val'
    file_val_lmdb = "gender_val_top3.list"
    convert_2_lmdb(Image_Root, file_val_lmdb, lmdb_path, save_image_size)
    
    file_train_lmdb = "gender_train_top3.list"
    lmdb_path = '/home/chenriwei/LMDB/OnlineImage/gender/margin_0.5_128_top3/train'
    convert_2_lmdb(Image_Root, file_train_lmdb, lmdb_path, save_image_size)
    
    caculate_image_mean(lmdb_path, save_path='/home/chenriwei/LMDB/OnlineImage/gender/margin_0.5_128_top3/mean.binaryproto')
