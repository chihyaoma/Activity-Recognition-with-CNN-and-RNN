# Simple optical flow algorithm

# python version: 2.7.6
# OpenCV version: 2.4.8


# Contact: Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
#          Chih-Yao Ma at cyma@gatech.edu
# Last update: 05/13/2016

# string: a = 'ZX'; b = 'CV'; a + b = 'ZXCV'
# list the files in a foler:
# 1. import os; os.listdir("./")
# 2. import glob; glob.glob('./*.avi')


import numpy as np
import cv2

cap = cv2.VideoCapture('v_Basketball_g01_c01.avi')

# information of the video
# property identifier:
# 1: ?; 2: s/frame; 3: width; 4: height; 6: ?; 7: ?
Fr = round(1 / cap.get(2))
#Fr = 25
Wd = int(cap.get(3))
Ht = int(cap.get(4))

# Define the codec and create VideoWriter object
# fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D') # opencv 2.4
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv 3.0
out = cv2.VideoWriter('out_flow.avi', fourcc, Fr, (Wd, Ht))

# read the first frame
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # convert to gray scale

# save in HSV (because of the optical flow algorithm we used)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while(cap.isOpened):
    # Capture frame-by-frame
    ret, frame2 = cap.read()

    if ret == True:
        # print "print frame.shape", frame.shape

        # compute the optical flow from two adjacent frames
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # show in RGB for visualization
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        frameProc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        out.write(frameProc)

        # Display the resulting frame
        cv2.imshow('Processed frame', frameProc)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prvs = next

    else:
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
