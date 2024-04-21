import cv2 as cv 
import cv2
import numpy as np

cap = cv.VideoCapture("calibration_videos/Camera2_Femi.MOV")

chessboard_size = (10,7)

objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

objpoints=[]
imgpoints=[]

i = 0

while True:
    ret,frame = cap.read()

    if i % 10 != 0:
        i += 1
        continue

    if not ret:
        break

    i += 1
    print(frame.shape)
    #frame = cv2.resize(frame, (1280,720), interpolation =cv2.INTER_AREA)

    gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

    #cv2.imshow('img', cv2.resize( gray, ( gray.shape[1]//3,gray.shape[0]//3 )))
    #cv2.waitKey(1)

    ret, corners = cv2.findChessboardCorners( gray, chessboard_size,  cv2.CALIB_USE_INTRINSIC_GUESS)

    if not ret:
        continue

    objpoints.append(objp)

    corners2 = cv2.cornerSubPix( gray, corners, (11,11), (-1,-1), (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001 ) )
    imgpoints.append(corners2)

    frame = cv2.resize( cv2.drawChessboardCorners( frame, chessboard_size, corners2, ret ), (frame.shape[1]//3,frame.shape[0]//3) )

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
