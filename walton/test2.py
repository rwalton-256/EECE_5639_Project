import cv2 as cv 
import cv2
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt

sobel_x = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])
sobel_y = np.array([
    [ 1, 2, 1],
    [ 0, 0, 0],
    [-1,-2,-1]
])

# These are the four corners of the fence I am centered in on the first frame,
# in order of top left, top right, bottom right, bottom left
pts1 = np.array( [
    [ 346, 118 ],
    [ 682, 120 ],
    [ 680, 324 ],
    [ 345, 311 ]
] )

pts2 = np.array( [
    [ 430, 379 ],
    [ 749, 392 ],
    [ 761, 577 ],
    [ 429, 587 ]
] )

m_int = np.load("m_int.npy")
print(m_int)
frames1 = np.load( "frames1.npy" )
frames2 = np.load( "frames2.npy" )

print(frames1.shape)

for i in range( 0, 300 ):

    frame1 = frames1[i,:,:,:]
    frame2 = frames2[i,:,:,:]

    plt.imshow(frame1[:,:,2])
    plt.show()

    plt.imshow(frame2[:,:,2])
    plt.show()
