# -*- coding: utf-8 -*-
import cv2
import numpy as np

def draw_rectangle(im, rect, color = (255,0,0)):
    '''
    rect = (x1,y1,x2,y2)
    '''
    cv2.rectangle(im, rect[0:2],rect[2:4], color = color, thickness = 1 )
    return im

def draw_landmarks(im, landmarks, color = (0,255,0)):
    '''
    landmarks = [(int,int),...,(int,int)]
    '''
    for landmark in landmarks:
        cv2.circle(im, landmark, color = color, thickness= -1, radius = 1 )
    return im

def draw_text(im, position, text, color = (0,0,255)):
    '''
    position = (x1,y1)
    text = "your text"
    '''
    cv2.putText(im, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 1 )
    return  im

def save_2_file(im, file_name):
    cv2.imwrite(file_name,im)

def read_image(filename):
    im = cv2.imread(filename)
    return im

