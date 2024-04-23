import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.spatial


def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]


def triangulate_points(P1, P2, x1, x2):
    """
    Two-view triangulation of points in
    x1,x2 (nx3 homog. coordinates).
    Similar to openCV triangulatePoints.
    """
    if not len(x2) == len(x1):
        raise ValueError("Number of points don't match.")
    X = [triangulate_nviews([P1, P2], [x[0], x[1]]) for x in zip(x1, x2)]
    return np.array(X)

pts = np.array( [
[2939,1360],
[2225,1494],
[3308,1372],
[2506,1488],
[3201,291],
[2300,591],
[3205,378],
[2300,665],
[3072,92],
[2216,417],
[3314,1891],
[2538,1922],
[2944,1858],
[2248,1930],
[2379,1051],
[1693,1234],
[1811,1910],
[1411,2111],
[3796,1740],
[2721,1771],
[2979,691],
[2139,917],
[3619,686],
[2578,929],
] )

pts1 = pts[0::2]
pts2 = pts[1::2]

pts1hom = np.pad(pts1,((0,0),(0,1)),'constant',constant_values=1)
pts2hom = np.pad(pts2,((0,0),(0,1)),'constant',constant_values=1)

ruler_pts = np.array( [
[1302,1053],
[578,1253],
[1944,1051],
[1290,1239],
] )

ruler_pts1 = ruler_pts[0::2]
ruler_pts2 = ruler_pts[1::2]

ruler_pts1hom = np.pad(ruler_pts1,((0,0),(0,1)),'constant',constant_values=1)
ruler_pts2hom = np.pad(ruler_pts2,((0,0),(0,1)),'constant',constant_values=1)

print(pts1hom)

def on_click(event):
    print(f"[{round(event.xdata)},{round(event.ydata)}],")

video1_path = "../Camera1_Femi_Walking.MOV"
cap1 = cv2.VideoCapture(video1_path)
fps1 = cap1.get(cv2.CAP_PROP_FPS)
image_width1, image_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

video2_path = "../Camera2_Femi_Walking.MOV"
cap2 = cv2.VideoCapture(video2_path)
fps2 = cap2.get(cv2.CAP_PROP_FPS)
image_width2, image_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

F, mask = cv2.findFundamentalMat(pts1,pts2)
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

print(F)
print(fps1)
print(fps2)

if 0:
    ret,im1 = cap1.read()
    ret,im2 = cap2.read()

    im1 = cv2.cvtColor( im1, cv2.COLOR_BGR2RGB )
    im2 = cv2.cvtColor( im2, cv2.COLOR_BGR2RGB )

    plt.subplot(121),plt.imshow(im1)
    plt.subplot(122),plt.imshow(im2)
    plt.connect('button_press_event',on_click)
    plt.show()

K1 = np.array( [
    [3.18019960e+03, 0.00000000e+00, 1.92428143e+03],
    [0.00000000e+00, 3.18131959e+03, 1.07018431e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
] )

K2 = np.array( [
    [3.19424770e+03, 0.00000000e+00, 1.92443412e+03],
    [0.00000000e+00, 3.19471817e+03, 1.09746668e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
] )

print(F)

E = K1.T @ F @ K2

print(E)

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
R = U @ W.T @ Vt
T = np.expand_dims( U[:,2], axis=0 )

T *= 0.1

data = np.zeros((1000,ruler_pts2hom.shape[0],3))
min_dist = np.inf
bets_t = None
for i in range(1000):
    T *= 1.01

    H1 = K1 @ np.concatenate( (np.eye(3), np.zeros((3,1))), axis=1 )
    H2 = K2 @ np.concatenate( (R,T.T), axis=1 )

    tr_hom = triangulate_points( H1, H2, ruler_pts1hom, ruler_pts2hom )

    dist = np.abs( np.linalg.norm( tr_hom[0] - tr_hom[1] ) - 3 )

    if dist < min_dist:
        min_dist = dist
        bets_t = T.copy()

    data[i,:,:] = tr_hom[:,0:3]

T = bets_t

H1 = K1 @ np.concatenate( (np.eye(3), np.zeros((3,1))), axis=1 )
H2 = K2 @ np.concatenate( (R,T.T), axis=1 )

print(R)
print(T)

print(H1)
print(H2)

print(bets_t)
print(min_dist)
ax = plt.figure().add_subplot(projection='3d')
for i in range(ruler_pts2hom.shape[0]):
    ax.scatter( data[:,i,0], data[:,i,1], data[:,i,2] )
plt.show()

while 1:
    ret,frame1 = cap1.read()
    ret,frame2 = cap2.read()

    Er = np.cross( np.eye(3), T ) @ R
    Fr = np.linalg.inv(K2).T @ Er @ np.linalg.inv(K1)

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(frame1,frame2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(frame2,frame1,lines2,pts2,pts1)

    cv2.namedWindow("Video 1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Video 2", cv2.WINDOW_NORMAL)
    cv2.imshow('Video 1', img3)
    cv2.imshow('Video 2', img5)
    cv2.resizeWindow("Video 1", (960,540))
    cv2.resizeWindow("Video 2", (960,540))
    key = cv2.waitKey()
    if key == 49:
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
    elif key == ord('q'):
        exit(0)
    elif key == ord('p'):

        im1 = cv2.cvtColor( frame1, cv2.COLOR_BGR2RGB )
        im2 = cv2.cvtColor( frame2, cv2.COLOR_BGR2RGB )
        plt.subplot(121),plt.imshow(im1)
        plt.subplot(122),plt.imshow(im2)
        plt.connect('button_press_event',on_click)
        plt.show()
    continue
    plt.subplot(121),plt.imshow(im1)
    plt.subplot(122),plt.imshow(im2)
    plt.connect('button_press_event',on_click)
    plt.show()

