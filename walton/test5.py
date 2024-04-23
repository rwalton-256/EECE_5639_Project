import cv2 as cv 
import cv2
import typing
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt

im1 = cv2.imread("data/ryans_iphone/roof_test/IMG_6734.jpg")
im2 = cv2.imread("data/ryans_iphone/roof_test/IMG_6735.jpg")

m_int = np.load("m_int.npy")
m_dist = np.load("m_dist.npy")
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(m_int, m_dist, (im1.shape[1],im1.shape[0]), 1, (im1.shape[1],im1.shape[0]))
x, y, w, h = roi

K = m_int

im1 = cv2.undistort(im1, m_int, m_dist, None, newcameramtx)
im2 = cv2.undistort(im2, m_int, m_dist, None, newcameramtx)

im1 = cv2.cvtColor( im1, cv2.COLOR_BGR2RGB )
im2 = cv2.cvtColor( im2, cv2.COLOR_BGR2RGB )

im1 = cv2.resize(im1,(im1.shape[1]//2,im1.shape[0]//2))
im2 = cv2.resize(im2,(im2.shape[1]//2,im2.shape[0]//2))

sift = cv2.SIFT.create()
bf = cv2.BFMatcher()

kp1,des1 = sift.detectAndCompute( im1, None )
print(len(kp1))
kp2,des2 = sift.detectAndCompute( im2, None )
print(len(kp2))

kp1 : typing.List[ cv2.KeyPoint ]
kp2 : typing.List[ cv2.KeyPoint ]

matches = bf.knnMatch( des1, des2, k=2 )
print(len(matches))

good = []
good : typing.List[ typing.List[ cv2.DMatch ] ]
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(
    im1, kp1,
    im2, kp2,
    good,
    None
)

print(kp1[0])

pts1 = []
pts2 = []
for match in good:
    pts1.append( kp1[match[0].queryIdx].pt )
    pts2.append( kp2[match[0].trainIdx].pt )
pts1 = np.array(pts1)
pts2 = np.array(pts2)

print(pts1)
print(pts2)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img",img3)
#cv2.resizeWindow("img",(img3))
key = cv2.waitKey()

print(len(good))
print(good)

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC,1,0.9)

print(F)
print(mask)
E = K.T @ F @ K

U,S,Vt = np.linalg.svd(E)

print( f"Fundamental matrix error: {np.abs(S[0]-S[1]) / np.mean(S[0:1])}" )

