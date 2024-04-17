import cv2 as cv 
import cv2
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt

frames1 = np.load( "frames1.npy" )
frames2 = np.load( "frames2.npy" )

sift = cv2.SIFT.create()
bf = cv2.BFMatcher()

for i in range( 0, 300 ):
    frame1 = ( frames1[i] * 256 ).astype( np.uint8 )
    frame2 = ( frames2[i] * 256 ).astype( np.uint8 )

    kp1,des1 = sift.detectAndCompute( frame1, None )
    kp2,des2 = sift.detectAndCompute( frame2, None )

    matches = bf.knnMatch( des1, des2, k=2 )

    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(
        frame1, kp1,
        frame2, kp2,
        good,
        None
    )

    plt.imshow(img3)
    plt.show()
