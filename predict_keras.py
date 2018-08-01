# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:34:16 2018

@author: yy
"""

import cv2
import sys
import gc
from train_keras import Model

if __name__ == '__main__':
    camera_idx = 0 #摄像头设备索引
    
    if len(sys.argv) != 1:
        print("ERROR:%s \r\n" % (sys.argv[0]))
        sys.exit(0)
        
    #加载模型
    model = Model()
    model.load_model(file_path = './model/face_model')    
              
    #框住人脸的矩形边框颜色       
    color = (0, 255, 0)
    
    #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(camera_idx)
    
    #人脸识别分类器本地存储路径
    cascade_path = "C:/Users/yy/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"
    
    #循环检测识别人脸
    while True:
        _, photo = cap.read()   #读取一帧视频
        
        #图像灰化，降低计算复杂度
        photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        
        #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)                

        #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(photo_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                
                #截取脸部图像提交给模型识别这是谁
                image = photo[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)   
#                print("face id --:",faceID)
                tag = 'XiongGan'
                if faceID == 0:                                                        
                    tag = 'XiongGan'
                elif faceID == 1:                                                        
                    tag = 'JingRu'
                elif faceID == 2:                                                        
                    tag = 'ZZZZZZ'
                else:
                    pass
        
                cv2.rectangle(photo, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    
                #文字提示是谁, 坐标 字体 字号 颜色 字的线宽
                cv2.putText(photo,tag, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)  
                                                      
        cv2.imshow("Surprise", photo)
        
        k = cv2.waitKey(10)
        #按q退出窗口，注意opencv 不支持窗口的关闭按钮关闭窗口
        if k & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()