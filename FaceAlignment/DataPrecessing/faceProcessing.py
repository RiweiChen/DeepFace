# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 16:36:33 2015

@author: RiweiChen <riwei.chen@outlook.com>
"""
import skimage
import skimage.io
import numpy as np
import cv2

def face_draw_point(filelist,savePath):
    '''
    人脸的剪切
    '''
    fid = open(filelist)
    lines = fid.readlines()
    fid.close()
    for line in lines:
        words = line.split(' ')
        filename = words[0]
        #im=skimage.io.imread(filename)
        im=cv2.imread(filename)
        #保存人脸的点，需要经过转换
        point = np.zeros((10,))
        
        point[0]=float(words[5])
        point[1]=float(words[6])
        point[2]=float(words[7])
        point[3]=float(words[8])
        point[4]=float(words[9])
        point[5]=float(words[10])
        point[6]=float(words[11])
        point[7]=float(words[12])
        point[8]=float(words[13])
        point[9]=float(words[14])
        
        for i in range(0,10,2):
            #skimage.draw.circle(point[i+1],point[i])
            cv2.circle(im, (int(point[i]),int(point[i+1])), 5, [0,0,255])            
            
        #skimage.io.imsave(savePath+filename,imcrop)
        cv2.imwrite(savePath+filename,im)
        #print words[0]
        #print words[1]
        

def face_prepare(filelist,fileout,savePath,w,h):
    '''
    人脸的剪切
    
    @debug： 谢谢网友@cyq0122 指出图像镜像过后，左右眼睛和嘴角都需要跟着改变的大BUG
    '''
    fid = open(filelist)
    lines = fid.readlines()
    fid.close()
    
    fid = open(fileout,'w')
    
    count = 1
    for line in lines:
        words = line.split(' ')
        filename = words[0]
        #im=skimage.io.imread(filename)
        im=cv2.imread(filename)
        print np.shape(im)
        x1 = int(words[1])
        y1 = int(words[2])
        x2 = int(words[3])
        y2 = int(words[4])
        #缩放的比例
        rate = float(y1-x1)/w
        #imcrop = im[x2:y2,x1:y1,:]
        imcrop = im[x2:y2,x1:y1,:]
        print np.shape(imcrop)
        #保存人脸的点，需要经过转换
        point = np.zeros((10,))
        
        point[0]=(float(words[5])-x1)/rate
        point[1]=(float(words[6])-x2)/rate
        point[2]=(float(words[7])-x1)/rate
        point[3]=(float(words[8])-x2)/rate
        point[4]=(float(words[9])-x1)/rate
        point[5]=(float(words[10])-x2)/rate
        point[6]=(float(words[11])-x1)/rate
        point[7]=(float(words[12])-x2)/rate
        point[8]=(float(words[13])-x1)/rate
        point[9]=(float(words[14])-x2)/rate
        imcrop = cv2.resize(imcrop,(w,h))
        #原始图像
        fid.write(str(count)+'.jpg')
        for i in range(0,10,2):
            #cv2.circle(imcrop, (int(point[i]),int(point[i+1])), 5, [0,0,255])  
            fid.write('\t'+str(point[i]))
            fid.write('\t'+str(point[i+1]))
        fid.write('\n')  
        
        # 翻转图像
        # @cyq0122 指出，左右眼睛需要交换，嘴巴也一样。
        imcrop_flip = cv2.flip(imcrop,1)
        fid.write(str(count)+'_flip.jpg')      
        fid.write('\t'+str(w-point[2]-1))
        fid.write('\t'+str(point[3]))
        fid.write('\t'+str(w-point[0]-1))
        fid.write('\t'+str(point[1]))
        fid.write('\t'+str(w-point[4]-1))
        fid.write('\t'+str(point[5]))
        fid.write('\t'+str(w-point[8]-1))
        fid.write('\t'+str(point[9]))
        fid.write('\t'+str(w-point[6]-1))
        fid.write('\t'+str(point[7]))
        fid.write('\n') 
        #skimage.io.imsave(savePath+filename,imcrop)
        #cv2.imwrite(savePath+filename,imcrop)
        cv2.imwrite(savePath+str(count)+'_flip.jpg',imcrop_flip)
        cv2.imwrite(savePath+str(count)+'.jpg',imcrop)
        count = count + 1
    fid.close()
        #print words[0]
        #print words[1]
        


if __name__ == "__main__":
    #train
    w = 39
    h = 39
    filelist=r"F:\Dataset\FacePoints\train\trainImageList.txt"
    filelistesave = r"F:\\MyDataset\\FacePoint\\train39X39\\train.list"
    savePath='F:\\MyDataset\\FacePoint\\train39X39\\'

    face_prepare(filelist,filelistesave,savePath,w,h)
    filelist=r"F:\Dataset\FacePoints\train\testImageList.txt"
    filelistesave = r"F:\\MyDataset\\FacePoint\\test39X39\\test.list"
    savePath='F:\\MyDataset\\FacePoint\\test39X39\\'
    face_prepare(filelist,filelistesave,savePath,w,h)
    #face_draw_point(filelist,savePath)
