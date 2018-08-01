# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:34:16 2018

@author: yy
"""
import sys
import cv2
import os
def photo2image(input_path, output_path):    
    num = 0 
    
    for dir_item in os.listdir(input_path):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(input_path, dir_item))
        
        if os.path.isdir(full_path):    
            #如果是文件夹，继续递归调用
            photo2image(full_path, output_path)
        else:   #文件 cv2.imread 能自动识别格式
            if dir_item.endswith('.jpg') or dir_item.endswith('.bmp'):
                #cv2.imread 能自动识别格式
                photo = cv2.imread(full_path)                
                
                #人脸识别分类器本地存储路径
                cascade_path = "C:/Users/yy/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"
                #告诉OpenCV使用人脸识别分类器
                classfier = cv2.CascadeClassifier(cascade_path)
                
                #将图像转换成灰度图像,opencv 人脸检测是基于灰度的
                grey = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)              
                    
                #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
                faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
                if len(faceRects) > 0:          #大于0则检测到人脸 ,一张图片中可能有多张人脸                                 
                    for faceRect in faceRects:  #遍历每一张人脸
                        x, y, w, h = faceRect                        
                        
                        #将当前帧保存为图片
                        cv2.imwrite('%s/%d.jpg'%(output_path, num), photo[y - 10: y + h + 10, x - 10: x + w + 10])  
                        num += 1
#                        print("save image:",full_path)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("ERROR:%s input_path, output_path\r\n" % (sys.argv[0]))
    else:
        photo2image(sys.argv[1], sys.argv[2])