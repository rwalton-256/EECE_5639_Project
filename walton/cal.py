import cv2 as cv 
import cv2
import numpy as np

cap = cv.VideoCapture("IMG_6679.MOV") 

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

objpoints=[]
imgpoints=[]

i = 0

while True:
    ret,frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (1280,720), interpolation =cv2.INTER_AREA)

    gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

    ret, corners = cv2.findChessboardCorners( gray, (6,9), None )

    if not ret:
        continue

    i += 1
    if i % 2:
        continue

    objpoints.append(objp)

    corners2 = cv2.cornerSubPix( gray, corners, (11,11), (-1,-1), (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001 ) )
    imgpoints.append(corners2)

    frame = cv2.resize( cv2.drawChessboardCorners( frame, (6,9), corners2, ret ), (frame.shape[1]//3,frame.shape[0]//3) )

    cv2.imshow('img', frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()

print("Starting calibration calc...")

ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(
    objectPoints=objpoints,
    imagePoints=imgpoints,
    imageSize=gray.shape[::-1],
    cameraMatrix=None,
    distCoeffs=None
)

print(ret)
print(mtx)
print(dist)

np.save( "m_int.npy", np.array(mtx) )
np.save( "m_dist.npy", np.array(dist) )
