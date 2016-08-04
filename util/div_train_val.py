# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:15:21 2015
@brief: 用与划分训练数据，train.list 和 val.list
@author: Riwei Chen <Riwei.Chen@outlook.com>
"""
import  os
def div_database(filepath,savepath,top_num=1000,equal_num=False,full_path=False):
    '''
    @brief: 提取webface人脸数据
    @param: filepath 文件路径
    @param: top_num=1000,表示提取的类别数目
    @param: equal_num 是否强制每个人都相同
    '''
    dirlists=os.listdir(filepath)
    dict_id_num={}
    for subdir in dirlists:
        dict_id_num[subdir]=len(os.listdir(os.path.join(filepath,subdir)))
    #sorted(dict_id_num.items, key=lambda dict_id_num:dict_id_num[1])
    sorted_num_id=sorted([(v, k) for k, v in dict_id_num.items()], reverse=True)
    select_ids=sorted_num_id[0:top_num]
    if equal_num == True:
        trainfile=save_path+'train_'+str(top_num)+'_equal.list'
        testfile=save_path+'val_'+str(top_num)+'_qeual.list'
    else:
        trainfile=save_path+'train_'+str(top_num)+'.list'
        testfile=save_path+'val_'+str(top_num)+'.list'
    fid_train=open(trainfile,'w')
    fid_test=open(testfile,'w')
    pid=0
    pre = ""
    if full_path ==True:
        pre = data_path
    for  select_id in select_ids:
        subdir=select_id[1]
        filenamelist=os.listdir(os.path.join(filepath,subdir)) 
        num=1
        for filename in filenamelist :
            #print select_ids[top_num-1]
            if equal_num==True and num>select_ids[top_num-1][0]:
                break
            if num%10!=0:
                fid_train.write(os.path.join(pre,subdir,filename)+'\t'+str(pid)+'\n')
            else:
                fid_test.write(os.path.join(pre,subdir,filename)+'\t'+str(pid)+'\n')
            num=num+1
        pid=pid+1
    fid_train.close()
    fid_test.close()

if __name__=='__main__':
    data_path = '/media/crw/MyBook/MyDataset/FaceDetection/crop_images'
    save_path = '/media/crw/MyBook/TrainData/ImageList/FaceDetection/aflw/'
    div_database(data_path,save_path, top_num=2, equal_num=False,full_path =True)
