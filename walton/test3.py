import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.spatial

K = np.array( [
    [ 1800,    0,  960 ],
    [    0, 1800,  540 ],
    [    0,    0,    1 ]
] )

# Uncalibrated points locations
pts1 = np.array( [
    [519,469], # Frame 0 
    [1020,486], # Frame 0 
    [1638,502], # Frame 0 
    [601,481], # Frame 0 
    [717,484], # Frame 15 
    [834,317], # Frame 42 
    [148,166], # Frame 163 
    [763,201], # Frame 30 
    [126,470], # Frame 269 
    [520,179], # Frame 0 
    [1025,182], # Frame 0 
    [11,197], # Frame 280 
    [1554,469], # Frame 1 
    [1873,433], # Frame 0 
    [130,366] # Frame 163 
] )

pts2 = np.array( [
    [643,881], # Frame 0 
    [1141,867], # Frame 0 
    [1578,855], # Frame 0 
    [837,883], # Frame 0 
    [949,880], # Frame 15 
    [1035,710], # Frame 42 
    [333,539], # Frame 163 
    [1064,602], # Frame 30 
    [382,904], # Frame 269 
    [646,568], # Frame 0 
    [1026,585], # Frame 0 
    [438,573], # Frame 280 
    [1007,844], # Frame 1 
    [1289,801], # Frame 0 
    [273,775] # Frame 163 
] )

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
print(F)
U,S,Vt = np.linalg.svd(F)
print(S)
#print(mask)

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
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

i = 0

frames1 = np.load( "frames1.npy" )
frames2 = np.load( "frames2.npy" )

while True:
    frame1 = frames1[i,:,:]
    frame2 = frames2[i,:,:]

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(frame1,frame2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(frame2,frame1,lines2,pts2,pts1)

    cv2.imshow("img1",cv2.resize(img3,(img3.shape[1]//2,img3.shape[0]//2)))
    cv2.imshow("img2",cv2.resize(img5,(img5.shape[1]//2,img5.shape[0]//2)))
    key = cv2.waitKey()

    # Left Arrow
    if key == 83:
        i += 1
        continue
    # Right Arrow
    elif key == 81:
        i -= 1
        continue
    # q
    elif key == 113:
        break
    elif key == 115:
        def on_click(event):
            print(f"[{round(event.xdata)},{round(event.ydata)}], # Frame {i} ")
        cv2.destroyAllWindows()
        plt.subplot(121),plt.imshow(img5)
        plt.subplot(122),plt.imshow(img3)
        plt.title(f"Frame {i}")
        plt.connect('button_press_event',on_click)
        plt.show()
        pass
    print(key)

E = K.T @ F @ K

E=-E
foo = ( E.T @ E ) / np.trace( E.T @ E )
print(foo)

bar = np.zeros((3))
for i in range(3):
    bar[i] = (1-foo[i,i])**2
#bar = -bar
print(bar)
print(np.linalg.norm(bar))

W = np.zeros((3,3))

W[0] = np.cross( E[0], bar )
W[1] = np.cross( E[1], bar )
W[2] = np.cross( E[2], bar )

rr = np.zeros((3,3))

rr[0,:] = W[0,:] + np.cross( W[1,:], W[2,:] )
rr[1,:] = W[1,:] + np.cross( W[2,:], W[0,:] )
rr[2,:] = W[2,:] + np.cross( W[0,:], W[1,:] )

print(rr)

print(scipy.spatial.transform.Rotation.from_matrix(rr).as_euler("zyx",degrees=True))
print(scipy.spatial.transform.Rotation.from_matrix(rr).as_matrix())

U,S,Vt = np.linalg.svd(E)

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
#R = U @ W.T @ Vt
T = U[:,2].T

#T = bar
#R = rr

frames1 = np.load( "frames1.npy" )
frames2 = np.load( "frames2.npy" )

while True:
    frame1 = frames1[i,:,:].copy()
    frame2 = frames2[i,:,:].copy()

    Er = np.cross( np.eye(3), T ) @ R
    Fr = np.linalg.inv(K).T @ Er @ np.linalg.inv(K)

    print(scipy.spatial.transform.Rotation.from_matrix(R).as_euler("zyx",degrees=True))

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,Fr)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(frame1,frame2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,Fr)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(frame2,frame1,lines2,pts2,pts1)

    cv2.imshow("img1",cv2.resize(img3,(img3.shape[1]//2,img3.shape[0]//2)))
    cv2.imshow("img2",cv2.resize(img5,(img5.shape[1]//2,img5.shape[0]//2)))
    key = cv2.waitKey()

    # Left Arrow
    if key == 83:
        i += 1
        continue
    # Right Arrow
    elif key == 81:
        i -= 1
        continue
    # q
    elif key == 113:
        break
    elif key == 49:
        R = R @ scipy.spatial.transform.Rotation.from_rotvec( np.pi / 1000 * np.array([1,0,0]) ).as_matrix()
    elif key == 50:
        R = R @ scipy.spatial.transform.Rotation.from_rotvec( -np.pi / 1000 * np.array([1,0,0]) ).as_matrix()
    elif key == 51:
        R = R @ scipy.spatial.transform.Rotation.from_rotvec( np.pi / 1000 * np.array([0,1,0]) ).as_matrix()
    elif key == 52:
        R = R @ scipy.spatial.transform.Rotation.from_rotvec( -np.pi / 1000 * np.array([0,1,0]) ).as_matrix()
    elif key == 53:
        R = R @ scipy.spatial.transform.Rotation.from_rotvec( np.pi / 1000 * np.array([0,0,1]) ).as_matrix()
    elif key == 54:
        R = R @ scipy.spatial.transform.Rotation.from_rotvec( -np.pi / 1000 * np.array([0,0,1]) ).as_matrix()
    elif key == 115:
        def on_click(event):
            print(f"[{round(event.xdata)},{round(event.ydata)}], # Frame {i} ")
        cv2.destroyAllWindows()
        plt.subplot(121),plt.imshow(img5)
        plt.subplot(122),plt.imshow(img3)
        plt.title(f"Frame {i}")
        plt.connect('button_press_event',on_click)
        plt.show()
        pass
    print(key)
