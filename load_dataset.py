# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:34:16 2018

@author: yy
"""

import sys
import cv2
import os
import numpy as np
IMAGE_SIZE = 64

#按照指定图像大小调整尺寸
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #获取图像尺寸
    h, w, _ = image.shape
    
    #对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)    
    
    #计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    #BGR颜色
    BLACK = [0, 0, 0]
    
    #给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    #调整图像大小并返回
    return cv2.resize(constant, (height, width))

#读取训练数据
images = []
labels = []
def read_path(data_path):    
    for dir_item in os.listdir(data_path):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(data_path, dir_item))
        
        if os.path.isdir(full_path):    
            #如果是文件夹，继续递归调用
            read_path(full_path)
        else:   #文件 cv2.imread 能自动识别格式
            if dir_item.endswith('.jpg') or dir_item.endswith('.bmp'):
                #cv2.imread 能自动识别格式
                image = cv2.imread(full_path)                
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                
                #放开这个代码，可以看到resize_image()函数的实际调用效果
                #cv2.imwrite('1.jpg', image)
                
                images.append(image)                
                labels.append(data_path)                                
                    
    return images,labels
    

#从指定路径读取训练数据
def load_dataset(data_path):
    images,labels = read_path(data_path)    
    
    #将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    #训练集图片总数 num，IMAGE_SIZE为64，故尺寸为 num * 64 * 64 * 3
    #图片为64 * 64像素,一个像素3个颜色值(BGR)
    images = np.array(images)
    print(images.shape)    
    
    #标注数据，三个文件夹的人分别标注为 0、1、2
    newlabels = []
    for label in labels:
        if label.endswith('xionggan'):
            newlabels.append(0)
        elif label.endswith('jingru'):
            newlabels.append(1)
        else:
            newlabels.append(2)
    labels = np.array(newlabels)
    
    return images, labels

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("ERROR:%s data_path\r\n" % (sys.argv[0]))    
    else:
        images, labels = load_dataset(sys.argv[1])