import cv2 as cv 
import cv2
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt

def get_gaussian_kernel( n : int, sigma : float ):

    gaussian_kernel = np.zeros( (n,n) )

    for i in range( gaussian_kernel.shape[0] ):
        ii = i - gaussian_kernel.shape[0] // 2
        for j in range( gaussian_kernel.shape[1] ):
            jj = j - gaussian_kernel.shape[1] // 2

            gaussian_kernel[i,j] = (
                1 / ( sigma * np.sqrt( 2 * np.pi ) ) *
                np.exp(
                    -0.5 * ( ii ** 2 + jj ** 2 ) /
                    ( sigma ** 2 )
                )
            )
    
    return gaussian_kernel

def convolve_2d( in_im : np.ndarray, in_filt : np.ndarray ) -> np.ndarray:
    n = in_filt.shape[0]
    assert n == in_filt.shape[1]
    ret = np.zeros((
        in_im.shape[0] - (n-1),
        in_im.shape[1] - (n-1)
    ))

    for i in range( ret.shape[0] ):
        for j in range( ret.shape[1] ):
            ret[i,j] = np.sum(
                ( in_filt * in_im[i:i+n,j:j+n] ).flatten()
            )

    return ret

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

foo = np.array( [
    [ 1, 1, 1, 1, 1 ],
    [ 1, 1, 1, 1, 1 ],
    [ 1, 1, 1, 1, 1 ],
    [ 1, 1, 1, 1, 1 ],
    [ 1, 1, 1, 1, 1 ]
] )
foo = foo / np.sum( foo )

frames1 = np.load( "frames1.npy" )
frames2 = np.load( "frames2.npy" )

dog = get_gaussian_kernel( 15, 2 ) - get_gaussian_kernel( 15, 6 )

#plt.imshow(dog)
#plt.colorbar()
#plt.show()

frame1_avg = frames1[0,:,:,:]
frame2_avg = frames2[0,:,:,:]

print(frame1_avg.shape)

for i in range( 0, 300 ):

    frame1 = frames1[i,:,:,:]
    frame2 = frames2[i,:,:,:]

    frame1_l = frames1[i-1,:,:,:]
    frame2_l = frames1[i-1,:,:,:]

    diff1 = np.linalg.norm( frame1 - frame1_avg, axis=-1 )
    diff2 = np.linalg.norm( frame2 - frame2_avg, axis=-1 )

    diff1 = scipy.signal.convolve2d( diff1, get_gaussian_kernel( 7, 3 ) )
    diff2 = scipy.signal.convolve2d( diff2, get_gaussian_kernel( 7, 3 ) )

    diff1[diff1 == 0] = 1e-9
    diff2[diff2 == 0] = 1e-9

    diff1 = 20 * np.log( diff1 )
    diff2 = 20 * np.log( diff2 )

    diff1 = ( diff1 - np.max( diff1 ) ) / 30 + 1
    diff2 = ( diff2 - np.max( diff2 ) ) / 30 + 1

    diff1 = np.clip( diff1, a_min=0, a_max=1 )
    diff2 = np.clip( diff2, a_min=0, a_max=1 )

    diff1[np.where(diff1>0.8)] = 1
    diff1[np.where(diff1<=0.8)] = 0
    diff2[np.where(diff2>0.8)] = 1
    diff2[np.where(diff2<=0.8)] = 0

    cv2.namedWindow("Camera 1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Camera 2", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Diff 1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Diff 2", cv2.WINDOW_NORMAL)

    cv2.imshow("Camera 1",frame1)
    cv2.imshow("Camera 2",frame2)
    cv2.imshow("Diff 1",diff1)
    cv2.imshow("Diff 2",diff2)

    cv2.resizeWindow("Camera 1", (640,360))
    cv2.resizeWindow("Camera 2", (640,360))
    cv2.resizeWindow("Diff 1", (640,360))
    cv2.resizeWindow("Diff 2", (640,360))

    frame1_avg = frame1_avg * 0.98
    frame1_avg += 0.02 * frame1
    frame2_avg = frame2_avg * 0.98
    frame2_avg += 0.02 * frame2

    key = cv.waitKey(1000000) & 0xFF
    if key == ord('q'): 
        exit(0)
    elif key == ord('s'): 
        plt.imshow(diff1)
        plt.colorbar()
        plt.show()

