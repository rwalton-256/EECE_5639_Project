import cv2
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt

K = np.array( [
    [ 1800,    0,  640 ],
    [    0, 1800,  360 ],
    [    0,    0,    1 ]
] )

# Top left corner
# Top right corner
# Bottom right corner
# Bottom left corner
# Far right top corner
# Far right bottom corner
# Right foot toe
# Nose
# Ball (NOTE The cameras are not time synched so this will be different!)

pts1 = np.array( [
    [ 346, 118 ],
    [ 682, 120 ],
    [ 680, 324 ],
    [ 345, 311 ],
    [ 1093, 116 ],
    [ 1092, 334 ],
    [ 524, 317 ],
    [ 497, 121 ],
    [ 502, 178 ]
] )

pts2 = np.array( [
    [ 430, 379 ],
    [ 749, 392 ],
    [ 761, 577 ],
    [ 429, 587 ],
    [ 1032, 401 ],
    [ 1051, 570 ],
    [ 689, 581 ],
    [ 655, 387 ],
    [ 697, 440 ]
] )

# Disregard the last point, it's the ball
F, mask = cv2.findFundamentalMat(pts1[0:-1,:],pts2[0:-1,:],method=cv2.FM_8POINT)
print(mask)

print(F)

H = K.T @ F @ K

print(H)

U,S,V = np.linalg.svd(H)

print(U)
print(S)
print(V)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple((np.random.randint(0,255,3)/256).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

frames1 = np.load( "frames1.npy" )
frames2 = np.load( "frames2.npy" )
frame1 = frames1[0,:,:,2]
frame2 = frames2[0,:,:,2]

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(frame1,frame2,lines1,pts1,pts2)
 
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(frame2,frame1,lines2,pts2,pts1)
 
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
