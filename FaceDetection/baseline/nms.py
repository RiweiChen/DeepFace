# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:16:26 2015

@author: crw
"""
import numpy as np
def nms_average(boxes, overlapThresh=0.2):
    '''
    @brief: 非极大值抑制取平均值
    '''
    if len(boxes) == 0:
        return []
    result_boxes = []
    # 初始化选择的索引
    pick = []
    # 获取窗口的坐标，是一个向量
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # 计算窗口的面积，并排序
    # 排序的依据是根据右下角的坐标？ 根据概率
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:,4])

    # 循环运行，直到没有索引
    while len(idxs) > 0:
        # 获取最后一个
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        #找出窗口的最大的坐标位置，以及最小的坐标位置。
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

         #计算窗口的长和宽
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # 计算第i个的面积
        area_i = np.maximum(0, x2[i] - x1[i] + 1) * np.maximum(0, y2[i] - y1[i] + 1)
        area_array = np.zeros(len(idxs) - 1)
        area_array.fill(area_i)
        #计算重合率
	overlap = (w * h) / (area[idxs[:last]])
	delete_idxs = np.concatenate(([last],np.where(overlap > overlapThresh)[0]))
        xmin = 10000
	ymin = 10000
	xmax = 0
	ymax = 0
	ave_prob  = 0
	width = x2[i] - x1[i] + 1
	height = y2[i] - y1[i] + 1
	for idx in delete_idxs:
		ave_prob += boxes[idxs[idx]][4]
		if(boxes[idxs[idx]][0] < xmin):
			xmin = boxes[idxs[idx]][0]
		if(boxes[idxs[idx]][1] < ymin):
			ymin = boxes[idxs[idx]][1]
		if(boxes[idxs[idx]][2] > xmax):
			xmax = boxes[idxs[idx]][2]
		if(boxes[idxs[idx]][3] > ymax):
			ymax = boxes[idxs[idx]][3]
	if(x1[i] - xmin >  0.1 * width):
		xmin = x1[i] - 0.1 * width
	if(y1[i] - ymin > 0.1 * height):
		ymin = y1[i] - 0.1 * height
	if(xmax - x2[i]> 0.1 * width):
		xmax = x2[i]  + 0.1 * width
	if( ymax - y2[i] > 0.1 * height):
		ymax = y2[i] + 0.1 * height
	result_boxes.append([xmin, ymin, xmax, ymax, ave_prob / len(delete_idxs)])
	# delete all indexes from the index list that have
        idxs = np.delete(idxs, delete_idxs)
    return result_boxes

def nms_max(boxes, overlapThresh=0.3):
    '''
    @brief: 对窗口进行非极大值抑制处理
    '''
    if len(boxes) == 0:
        return []
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:,4])
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        area_i = np.maximum(0, x2[i] - x1[i] + 1) * np.maximum(0, y2[i] - y1[i] + 1)
        area_array = np.zeros(len(idxs) - 1)
        area_array.fill(area_i)
        overlap = (w * h) / (area[idxs[:last]]  - w * h + area_array)
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
    return boxes[pick]

