# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:34:16 2018

@author: yy
"""
import sys
import cv2

def captureFaceFromVideo(face_num_max, output_path):
    window_name = 'Face Capture'
    camera_idx = 0 #摄像头设备索引
    
    cv2.namedWindow(window_name)
    #视频来源，摄像头
    cap = cv2.VideoCapture(camera_idx)                
    
    #人脸识别分类器本地存储路径
    cascade_path = "C:/Users/yy/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"
    #告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier(cascade_path)
    
    #识别出人脸后要画的边框的颜色，opencv 用的是BGR格式
    color = (0, 255, 0)
    
    num = 0    
    while cap.isOpened():
        ok, photo = cap.read() #读取一帧数据
        if not ok:            
            break    
            
        #将图像转换成灰度图像,opencv 人脸检测是基于灰度的
        grey = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)           
        
        #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:          #大于0则检测到人脸 ,一张图片中可能有多张人脸                                  
            for faceRect in faceRects:  #遍历每一张人脸
                x, y, w, h = faceRect                        
                
                #输出人脸图片
                cv2.imwrite('%s/%d.jpg'%(output_path, num), photo[y - 10: y + h + 10, x - 10: x + w + 10])                                
                                
                num += 1                
                if num > (face_num_max):   #如果超过指定最大保存数量退出循环
                    break
                
                #画出矩形框
                cv2.rectangle(photo, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                
                #在显示框中显示当前捕捉到了多少人脸图片，反馈用户
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(photo,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)                
        
        #超过指定最大保存数量结束程序
        if num > (face_num_max): break                
                       
        #显示图像
        cv2.imshow(window_name, photo)        
        c = cv2.waitKey(10)
        #按q退出窗口，注意opencv 不支持窗口的关闭按钮关闭窗口
        if c & 0xFF == ord('q'):
            break        
    
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows() 
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("ERROR:%s face_num_max output_path\r\n" % (sys.argv[0]))
    else:
        captureFaceFromVideo(int(sys.argv[1]), sys.argv[2])