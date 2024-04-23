import cv2
import numpy as np

cap = cv2.VideoCapture("data/ryans_iphone/calibration/IMG_6733.MOV") 

m_int = np.load("m_int.npy")
m_dist = np.load("m_dist.npy")
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(m_int, m_dist, (1080,1920), 1, (1080,1920))
x, y, w, h = roi

while True:
    ret,frame = cap.read()

    print(frame.shape)
    frame_und = cv2.undistort(frame, m_int, m_dist, None, newcameramtx)
    print(frame_und.shape)

    cv2.imshow('img', cv2.resize( frame, (1920//2,1080//2) ))
    cv2.imshow('img_und', cv2.resize( frame_und, (1920//2,1080//2) ))
    cv2.waitKey(1)
