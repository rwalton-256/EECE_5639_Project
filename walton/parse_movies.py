import cv2 as cv 
import cv2
import numpy as np

cap1 = cv.VideoCapture("IMG_6628.MOV") 

for i in range(1045):
    assert cap1.isOpened()
    ret,frame = cap1.read()

cap2 = cv.VideoCapture("IMG_2003.MOV") 

for i in range(1419):
    assert cap2.isOpened()
    ret,frame = cap2.read()

for i in range(126):
    ret,frame1 = cap1.read()
    frame1 = cv2.resize(frame1, (720,1280), interpolation =cv2.INTER_AREA)
    frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ret,frame2 = cap2.read()

frames1 = []
frames2 = []

m_int = np.load("m_int.npy")
m_dist = np.load("m_dist.npy")
newcameramtx, roi = cv.getOptimalNewCameraMatrix(m_int, m_dist, (720,1080), 1, (720,1080))
x, y, w, h = roi
print(newcameramtx)
print(roi)

try:
    while True:
        ret,frame1 = cap1.read()
        if not ret:
            raise ""
        frame1 = cv2.resize(frame1, (720,1280), interpolation =cv2.INTER_AREA)
        frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ret,frame2 = cap2.read()
        if not ret:
            raise ""

        #frame1 = cv.undistort(frame1, m_int, m_dist, None, newcameramtx)
        #frame2 = cv.undistort(frame2, m_int, m_dist, None, newcameramtx)

        frame1 = np.array(cv.cvtColor(frame1,cv.COLOR_RGB2HSV)).astype(np.float32)/256
        frame2 = np.array(cv.cvtColor(frame2,cv.COLOR_RGB2HSV)).astype(np.float32)/256
        #frame1 = np.array(frame1).astype(np.float32) / 256
        #frame2 = np.array(frame2).astype(np.float32) / 256

        frames1.append(frame1)
        frames2.append(frame2)

        print( len( frames1 ) )
except:
    frames1 = np.array( frames1 )
    frames2 = np.array( frames2 )

    num_frames = np.maximum( frames1.shape[0], frames2.shape[0] )

    frames1 = frames1[0:num_frames]
    frames2 = frames2[0:num_frames]

    np.save( "frames1.npy", frames1 )
    np.save( "frames2.npy", frames2 )


