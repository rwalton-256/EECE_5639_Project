import cv2 as cv 
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

cap1 = cv.VideoCapture("IMG_6628.MOV") 

for i in range(1045):
    assert cap1.isOpened()
    ret,frame = cap1.read()

cap2 = cv.VideoCapture("IMG_2003.MOV") 

for i in range(1419):
    assert cap2.isOpened()
    ret,frame = cap2.read()

for i in range(126):
    ret,frame1 = cap1.read()
    frame1 = cv2.resize(frame1, (720,1280), interpolation =cv2.INTER_AREA)
    frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ret,frame2 = cap2.read()

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

hist_len = 20
frame1_hist = np.zeros( ( hist_len, frame1.shape[0], frame1.shape[1] ) )
frame2_hist = np.zeros( ( hist_len, frame2.shape[0], frame2.shape[1] ) )

frame1_last = None
frame2_last = None
frame1_avg = None
frame2_avg = None
d_last = None

h1,status = cv2.findHomography( pts1, pts2 )
h2,status = cv2.findHomography( pts2, pts1 )
breakpoint()
while True:
    ret,frame1 = cap1.read()
    frame1 = cv2.resize(frame1, (720,1280), interpolation =cv2.INTER_AREA)
    frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ret,frame2 = cap2.read()

    frame1 = np.array(cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)).astype(np.double)/256
    frame2 = np.array(cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)).astype(np.double)/256

    frame1_med = np.median( frame1_hist, 0 )
    frame2_med = np.median( frame2_hist, 0 )

    #d = Gx + Gy * 1j

    diff1 = (frame1-frame1_med+1)/2
    diff2 = (frame2-frame2_med+1)/2

    #sq_size = 7
    #fft_size = sq_size * 2 - 1
    #ifft_size = fft_size * 3
    #res = np.zeros( (frame1.shape[0]-sq_size+1,frame1.shape[1]-sq_size+1), dtype=np.complex128 )

    #for row in range(res.shape[0]):
    #    for col in range(res.shape[1]):
    #        foo = np.fft.fftshift(
    #            np.fft.ifft2(
    #                           np.fft.fft2( d[row:row+sq_size,col:col+sq_size], [ fft_size ] * 2 )
    #                * np.conj( np.fft.fft2( d_last[row:row+sq_size,col:col+sq_size], [ fft_size ] * 2 ) ),
    #                [ ifft_size ] * 2
    #            )
    #        )
    #        print(f"{row},{col}")
    #        loc = ( np.array( np.unravel_index( np.argmax( np.abs(foo) ), foo.shape ) ) - ifft_size // 2 ) * sq_size / ifft_size
    #        res[row,col] = loc[0] + loc[1] * 1j

    #plt.imshow( np.real( res ) )
    #plt.show()
    #plt.imshow( np.imag( res ) )
    #plt.show()

    #flow = cv.calcOpticalFlowFarneback(frame1, frame1_last,
    #                                   None,
    #                                   0.5, 3, 5, 20, 5, 1.2, 0)

    #flow -= np.median(flow)
    #flow /= np.max([np.max(flow), -np.min(flow)]) * 2
    #flow += 0.5

    #print(np.max(flow))
    #print(np.min(flow))
    cv2.imshow("Camera 1",frame1)
    cv2.imshow("Camera 2",frame2)
    cv2.imshow("Diff 1",diff1)
    cv2.imshow("Diff 2",diff2)
    #cv2.imshow("Flowx",flow[:,:,0])
    #cv2.imshow("Flowy",flow[:,:,1])
    #frame1_warped = cv2.warpPerspective( frame1, h1, (frame1.shape[1],frame1.shape[0]))
    #frame2_warped = cv2.warpPerspective( frame2, h2, (frame1.shape[1],frame1.shape[0]))
    #cv2.imshow("Camera 1 in Camera 2 warp",frame1_warped)
    #cv2.imshow("Camera 2 in Camera 1 warp",frame2_warped)

    frame1_hist[1:,:,:] = frame1_hist[0:-1,:,:]
    frame2_hist[1:,:,:] = frame2_hist[0:-1,:,:]
    frame1_hist[0,:,:] = frame1
    frame2_hist[0,:,:] = frame2
    if cv.waitKey(1000000) & 0xFF == ord('q'): 
        exit(0)

# Converts frame to grayscale because we 
# only need the luminance channel for 
# detecting edges - less computationally  
# expensive 
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY) 
  
# Creates an image filled with zero 
# intensities with the same dimensions  
# as the frame 
mask = np.zeros_like(first_frame) 
  
# Sets image saturation to maximum 
mask[..., 1] = 255

while True:
    # Opens a new window and displays the input 
    # frame 
    cv.imshow("input", frame) 
      
    # Converts each frame to grayscale - we previously  
    # only converted the first frame to grayscale 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      
    # Calculates dense optical flow by Farneback method 
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray,  
                                       None, 
                                       0.5, 3, 15, 3, 5, 1.2, 0) 
      
    # Computes the magnitude and angle of the 2D vectors 
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1]) 

    # Sets image hue according to the optical flow  
    # direction 
    mask[..., 0] = angle * 180 / np.pi / 2
      
    # Sets image value according to the optical flow 
    # magnitude (normalized) 
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX) 
      
    # Converts HSV to RGB (BGR) color representation 
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR) 
      
    # Opens a new window and displays the output frame 
    cv.imshow("dense optical flow", rgb) 
      
    # Updates previous frame 
    prev_gray = gray 
      
    # Frames are read by intervals of 1 millisecond. The 
    # programs breaks out of the while loop when the 
    # user presses the 'q' key 
    if cv.waitKey(1000000) & 0xFF == ord('q'): 
        break
  
# The following frees up resources and 
# closes all windows 
cap.release() 
cv.destroyAllWindows() 