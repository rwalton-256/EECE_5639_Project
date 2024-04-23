import cv2 as cv 
import cv2
import numpy as np

cap1 = cv.VideoCapture("IMG_6628.MOV") 

for i in range(1045):
    assert cap1.isOpened()
    ret,frame = cap1.read()
print(frame.shape)

cap2 = cv.VideoCapture("IMG_2003.MOV") 

for i in range(1419):
    assert cap2.isOpened()
    ret,frame = cap2.read()
print(frame.shape)

for i in range(26+78):
    ret,frame1 = cap1.read()
    #frame1 = cv2.resize(frame1, (720,1280), interpolation =cv2.INTER_AREA)
    #frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ret,frame2 = cap2.read()

frames1 = []
frames2 = []

m_int = np.load("m_int.npy")
m_dist = np.load("m_dist.npy")
newcameramtx, roi = cv.getOptimalNewCameraMatrix(m_int, m_dist, (1080,1920), 1, (1080,1920))
x, y, w, h = roi
print(newcameramtx)
print(roi)

out_1 = cv.VideoWriter("Camera1.mp4", cv2.VideoWriter_fourcc(*'MJPG'), 30, (1920, 1080))
out_2 = cv.VideoWriter("Camera2.mp4", cv2.VideoWriter_fourcc(*'MJPG'), 30, (1920, 1080))

try:
    for i in range(287):
        ret,frame1 = cap1.read()
        if not ret:
            raise ""
        #frame1 = cv2.resize(frame1, (720,1280), interpolation =cv2.INTER_AREA)
        frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ret,frame2 = cap2.read()
        if not ret:
            raise ""

        #frame1 = np.array(frame1).astype(np.float32) / 256
        #frame2 = np.array(frame2).astype(np.float32) / 256
        frame2 = cv2.resize(frame2, (1920,1080), interpolation =cv2.INTER_AREA)

        out_1.write(frame1)
        out_2.write(frame2)

        frame1_und = cv.undistort(frame1, m_int, m_dist, None, newcameramtx)
        frame2_und = cv.undistort(frame2, m_int, m_dist, None, newcameramtx)

        frame1 = np.array(frame1)
        frame2 = np.array(frame2)

        frames1.append(frame1)
        frames2.append(frame2)

        print(frame1.shape)
        print(frame2.shape)
        print(frame1_und.shape)
        print(frame2_und.shape)

        cv2.imshow('img1', cv2.resize( frame1, (1920//2,1080//2) ))
        cv2.waitKey(1)
        cv2.imshow('img2', cv2.resize( frame2, (1920//2,1080//2) ))
        cv2.waitKey(1)

        cv2.imshow('img1_und', cv2.resize( frame1_und, (1920//2,1080//2) ))
        cv2.waitKey(1)
        cv2.imshow('img2_und', cv2.resize( frame2_und, (1920//2,1080//2) ))
        cv2.waitKey(1)

        print( len( frames1 ) )
finally:
    frames1 = np.array( frames1 )
    frames2 = np.array( frames2 )

    num_frames = np.maximum( frames1.shape[0], frames2.shape[0] )

    sframes = np.array(frames1)
    sframes = sframes[0:num_frames]
    np.save( "frames1.npy", sframes )
    sframes = np.array(frames2)
    sframes = sframes[0:num_frames]
    np.save( "frames2.npy", sframes )
