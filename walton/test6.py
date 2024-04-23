import cv2 as cv 
import cv2
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt

def local_maxima(array2d):
    return ((array2d > np.roll(array2d,  1, 0)) &
            (array2d > np.roll(array2d, -1, 0)) &
            (array2d > np.roll(array2d,  1, 1)) &
            (array2d > np.roll(array2d, -1, 1)))


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

F, mask = cv2.findFundamentalMat(pts1/4,pts2/4)

def get_large_obj_rejection_kernel( reject_sq_size : int, accept_circ_size : int ):
    assert( reject_sq_size > accept_circ_size )

    ret = np.full( (2*reject_sq_size+1,2*reject_sq_size+1), -1, dtype=float )

    for i in range( -reject_sq_size, reject_sq_size + 1 ):
        for j in range( -reject_sq_size, reject_sq_size + 1 ):
            norm_dist_from_cent = np.sqrt( i * i + j * j ) / accept_circ_size
            if norm_dist_from_cent < 1:
                parab_val = ( 1 - norm_dist_from_cent ) ** 2
                ret[i+reject_sq_size,j+reject_sq_size] = parab_val
    
    return ret

large_obj_reject_kernel = get_large_obj_rejection_kernel(12,9)
#plt.imshow(large_obj_reject_kernel)
#plt.colorbar()
#plt.show()

frames1 = np.load( "frames1.npy" )
frames2 = np.load( "frames2.npy" )

frames1 = frames1[200:,::4,::4,:]
frames2 = frames2[200:,::4,::4,:]

frame1_avg = np.mean( frames1[0:20,:,:,:], axis=0 )
frame2_avg = np.mean( frames2[0:20,:,:,:], axis=0 )

print(frame1_avg)
print(frame1_avg.shape)

for i in range( 0, 300 ):

    frame1 = frames1[i,:,:,:]
    frame2 = frames2[i,:,:,:]

    diff1 = np.linalg.norm( frame1 - frame1_avg, axis=-1 )
    diff2 = np.linalg.norm( frame2 - frame2_avg, axis=-1 )

    #plt.clf()
    #plt.imshow(diff1)
    #plt.colorbar()
    #plt.show()

    diff1 = scipy.signal.convolve2d( diff1, large_obj_reject_kernel, 'same' )
    diff2 = scipy.signal.convolve2d( diff2, large_obj_reject_kernel, 'same' )

    #plt.clf()
    #plt.imshow(diff1)
    #plt.colorbar()
    #plt.show()

    thresh = 0

    frame1_vis = frame1.copy()
    frame2_vis = frame2.copy()

    diff1[np.where(diff1<0)] = 0
    ind1 = np.where(local_maxima(diff1))
    diff1 = np.full_like(diff1,0)
    diff1[ind1] = 1
    if ind1[0].shape[0] != 0:
        ind1 = np.array(ind1).T
        lines1 = cv2.computeCorrespondEpilines( ind1[:,::-1], 1, F )
        lines1 = lines1.reshape(-1,3)
        for line in lines1:
            x0y0 = (0, int(-line[2]/line[1]))
            x1y1 = (frame2_vis.shape[1], int(-(line[2]+line[0]*frame2_vis.shape[1])//line[1]))
            frame2_vis = cv2.line( frame2_vis, x0y0, x1y1, (255,255,255), 1 )

    diff2[np.where(diff2<0)] = 0
    ind2 = np.where(local_maxima(diff2))
    diff2 = np.full_like(diff2,0)
    diff2[ind2] = 1
    if ind2[0].shape[0] != 0:
        ind2 = np.array(ind2).T
        lines2 = cv2.computeCorrespondEpilines( ind2[:,::-1], 2, F )
        lines2 = lines2.reshape(-1,3)
        for line in lines2:
            x0y0 = (0, int(-line[2]/line[1]))
            x1y1 = (frame1_vis.shape[1], int(-(line[2]+line[0]*frame1_vis.shape[1])//line[1]))
            frame1_vis = cv2.line( frame1_vis, x0y0, x1y1, (255,255,255), 1 )

    cv2.namedWindow("Camera 1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Camera 2", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Diff 1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Diff 2", cv2.WINDOW_NORMAL)

    cv2.imshow("Camera 1",frame1_vis)
    cv2.imshow("Camera 2",frame2_vis)
    cv2.imshow("Diff 1",diff1)
    cv2.imshow("Diff 2",diff2)

    cv2.resizeWindow("Camera 1", (960,540))
    cv2.resizeWindow("Camera 2", (960,540))
    cv2.resizeWindow("Diff 1", (960,540))
    cv2.resizeWindow("Diff 2", (960,540))
    cv2.resizeWindow("Avg 1", (960,540))
    cv2.resizeWindow("Avg 2", (960,540))

    frame1_avg = frame1_avg * 0.9
    frame1_avg += 0.1 * frame1
    frame2_avg = frame2_avg * 0.9
    frame2_avg += 0.1 * frame2

    key = cv.waitKey(500) & 0xFF
    if key == ord('q'): 
        exit(0)
    elif key == ord('s'): 
        plt.imshow(diff1)
        plt.colorbar()
        plt.show()

