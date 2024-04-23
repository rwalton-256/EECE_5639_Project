import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.spatial


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

def on_click(event):
    print(f"[{round(event.xdata)},{round(event.ydata)}],")
cv2.destroyAllWindows()
plt.subplot(121),plt.imshow(im1)
plt.subplot(122),plt.imshow(im2)
plt.connect('button_press_event',on_click)
plt.show()

pts = np.array( [
[1825,1478],
[2345,1383],
[1766,1425],
[2285,1324],
[1915,1564],
[2254,1459],
[1335,1568],
[1638,1398],
[1790,1297],
[2344,1201],
[1726,2027],
[1287,1776],
[1925,2010],
[1474,1791],
[2882,1849],
[2535,1795],
[2589,1527],
[2856,1487],
[718,1731],
[1068,1475],
[1065,1407],
[1441,1235],
[1273,1949],
[1252,1693],
[1433,659],
[2885,574],
[1151,844],
[2486,757],
[73,1306],
[1294,1124],
[558,585],
[1462,502],
[859,995],
[1466,869],
] )

# Uncalibrated points locations
pts1 = pts[0::2]
pts2 = pts[1::2]

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC,2,0.9999)
print(mask)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''
    r,c,cc = img1.shape
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    np.random.seed(1)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple((np.random.randint(0,255,3)).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,2)
        img1 = cv2.circle(img1,tuple(pt1),7,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),7,color,-1)
    return img1,img2

i = 0

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(im1.copy(),im2.copy(),lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(im2.copy(),im1.copy(),lines2,pts2,pts1)

cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
cv2.imshow("img1",img3)
cv2.imshow("img2",img5)
cv2.resizeWindow("img1",(img3.shape[1]//4,img3.shape[0]//4))
cv2.resizeWindow("img2",(img5.shape[1]//4,img5.shape[0]//4))
key = cv2.waitKey()

print(K)
E = K.T @ F @ K

U,S,Vt = np.linalg.svd(E)

print( f"Fundamental matrix error: {np.abs(S[0]-S[1]) / np.mean(S[0:1])}" )

W = np.array( [
    [ 0,-1, 0 ],
    [ 1, 0, 0 ],
    [ 0, 0, 1 ]
] )

if np.linalg.det(U) < 0:
    U *= -1

if np.linalg.det(Vt) < 0:
    Vt *= -1

R = U @ W @ Vt
T = U[:,2].T

Er = np.cross( np.eye(3), T ) @ R
Fr = np.linalg.inv(K).T @ Er @ np.linalg.inv(K)

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,Fr)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(im1.copy(),im2.copy(),lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,Fr)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(im2.copy(),im1.copy(),lines2,pts2,pts1)

cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
cv2.imshow("img1",img3)
cv2.imshow("img2",img5)
cv2.resizeWindow("img1",(img3.shape[1]//4,img3.shape[0]//4))
cv2.resizeWindow("img2",(img5.shape[1]//4,img5.shape[0]//4))
key = cv2.waitKey()
