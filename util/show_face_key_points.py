# -*- coding: utf-8 -*-
'''
@brief 可视化关键点
'''
import os
import matplotlib.pylab as plt
import cv2
import skimage.io as io
def show_image_with_point(img,points):
    for i in range(0,10,2):
        cv2.circle(img, (points[i],points[i+1]) ,2, (255,0,0))
    plt.imshow(img)
    io.imsave('result.jpg',img)
    
if __name__ =="__main__":
    file_path = "E:/Dataset/CelebA/Origin/"
    file_list = "E:/Dataset/CelebA/list_landmarks_celeba.txt"
    fid = open(file_list)
    fid.readline()
    fid.readline()
    lines = fid.readlines()
    fid.close()
    for line in lines:
        items = line.split()
        file_ = items[0]
        print "processing: ",file_
        file_name = os.path.join(file_path,file_)
        points = [int(items[i]) for i in range(1,11)]
        img = io.imread(file_name)
        show_image_with_point(img,points)
        break
        
        
