# Simple optical flow algorithm
#
#
# OpenCV version: 2.4.8
#
#
# Contact:  
# Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
# Chih-Yao Ma at <cyma@gatech.edu>
#
# Last update: 05/13/2016

import numpy as np
import cv2
from scripts.flownet import FlowNet


cap = cv2.VideoCapture('v_Basketball_g01_c01.avi')

# information of the video
# property identifier:
# 1: ?; 2: s/frame; 3: width; 4: height; 6: ?; 7: ?
Fr = int(round(1/cap.get(2)))
#Fr = 25
Wd = int(cap.get(3)) 
Ht = int(cap.get(4))

print Fr
print Wd
print Ht

# Define the codec and create VideoWriter object
# fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out_flow.avi',fourcc, Fr, (Wd,Ht))

# read the first frame
ret, frame1 = cap.read()
# prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # convert to gray scale
prvs = frame1

# save in HSV (because of the optical flow algorithm we used)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(cap.isOpened):
    # Capture frame-by-frame
    ret, frame2 = cap.read()

    if ret==True:
    	# Get frame sizes
        height, width, channels = prvs.shape

        next = frame2

        cv2.imshow('Frame 1',prvs)
        cv2.imshow('Frame 2',next)

        # save the frames into png files for FlowNet to read 
        # TODO: this maybe stupid but is the easiest way without reconfigure 
        # the FlowNet and possible re-train the model
        cv2.imwrite('data/frame1.png',prvs)
        cv2.imwrite('data/frame2.png',next)

    	# compute the optical flow from two adjacent frames
        FlowNet.run(prvs) # the FlowNet will save a .flo file




        # read the .flo file
        fileName = 'flownetc-pred-0000000.flo'
        flowMapSize = np.fromfile(fileName,np.float32, count = 1)
        if flowMapSize != 202021.25:
            print 'Dimension incorrect. Invalid .flo file'
        else:
            data = np.fromfile(fileName, np.float32, count = 2*width*height)

        flow = np.resize(data, (height, width, 2))

        # show in RGB for visualization
        # TODO: there is something wrong here...
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        frameProc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    	out.write(frameProc)

    	 # Display the resulting frame
    	cv2.imshow('Processed frame',frameProc)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prvs = next
        
    else:
    	break
    
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

