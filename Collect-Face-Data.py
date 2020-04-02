import  numpy as np
import cv2
from matplotlib import pyplot as plt
import os

directory='C:\\Users\\1000267332\\PycharmProjects\\OpenCV\\Practice\\Face-emotions\\Data\\test\\Yawning\\'
cap=cv2.VideoCapture(0)

while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    cv2.rectangle(frame, (175, 100), (500, 360), (0, 255, 0), 2)
    cv2.imshow('frame',frame)
    # plt.imshow(frame)
    # plt.show()
    roi=frame[100:360,175:500]
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    _,roi=cv2.threshold(roi,30, 255,cv2.THRESH_BINARY)
    kernel=np.ones((1,1),np.uint8)
    roi=cv2.dilate(roi,kernel,iterations=1)
    cv2.imshow('roi',roi)

    interupt=cv2.waitKey(10)

    if interupt & 0xFF==ord('0'):
        print('count '+str(len(os.listdir(directory))))
        cv2.imwrite(directory+str(len(os.listdir(directory)))+'.jpg',roi)
    if interupt & 0xFF==ord('1'):
        print('count '+str(len(os.listdir(directory))))
        cv2.imwrite(directory+str(len(os.listdir(directory)))+'.jpg',roi)
    if interupt & 0xFF==ord('2'):
        print('count '+str(len(os.listdir(directory))))
        cv2.imwrite(directory+str(len(os.listdir(directory)))+'.jpg',roi)

    if interupt & 0xff==ord('q'):
        print('break')
        break

cap.release()
cv2.destroyAllWindows()
