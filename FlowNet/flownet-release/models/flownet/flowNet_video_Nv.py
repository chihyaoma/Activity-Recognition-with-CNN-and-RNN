# FlowNet algorithm
# for one video

# python version: 2.7.6 / 3.4.3
# OpenCV version: 3.1.0

# Contact:
# Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
# Chih-Yao Ma at <cyma@gatech.edu>
#
# Last update: 05/28/2016

import numpy as np
import cv2
from scripts.flownet import FlowNet
from time import time

# read the video file

# cap = cv2.VideoCapture('v_Basketball_g01_c01.avi')
cap = cv2.VideoCapture('v_Archery_g01_c06.avi')


# information of the video
# Fr = round(1 / cap.get(2))  # frame rate
fps = int(cap.get(cv2.CAP_PROP_FPS))
Wd = int(cap.get(3))
Ht = int(cap.get(4))
nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get number of frames

# Define the codec and create VideoWriter object
# fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# initialize the display window
cv2.namedWindow('Previous, current frames and flow map')

# output named
filename = 'FlowNet_out_N.avi'

step = 5  # steps for computing optical flow

numFlowMap = int(nFrame / step)

# initialize video file with 1 FPS
out = cv2.VideoWriter(filename, fourcc, fps / step, (Wd, Ht))

# read the first frame
ret, prvs = cap.read()

# Get frame sizes
height, width, channels = prvs.shape

# save in HSV (because of the optical flow algorithm we used)
hsv = np.zeros((height, width, channels, numFlowMap))
hsv[:, :, 1, :] = 255

indFrame = 1
indFlowMap = 0
maxMag = 0

tStart = time()  # calculate the time difference

while(cap.isOpened):
    # Capture frame-by-frame
    ret, next = cap.read()

    if ret is True:
        indFrame = indFrame + 1

        if ((indFrame - 1) % step) == 0 and indFlowMap < numFlowMap:

            imgDisplay = np.hstack((prvs, next))

            # save the frames into png files for FlowNet to read
            # TODO: stupid but is the easiest way without reconfigure
            # the FlowNet and possible re-train the model
            cv2.imwrite('data/frame1.png', prvs)
            cv2.imwrite('data/frame2.png', next)

            # compute the optical flow from two adjacent frames
            FlowNet.run(prvs)  # the FlowNet will save a .flo file

            # read the .flo file
            fileName = 'flownetc-pred-0000000.flo'
            flowMapSize = np.fromfile(fileName, np.float32, count=1)
            if flowMapSize != 202021.25:
                print 'Dimension incorrect. Invalid .flo file'
            else:
                data = np.fromfile(fileName, np.float32,
                                   count=2 * width * height)
            flow = np.resize(data, (height, width, 2))

            # prune the flow value if it's weirdly large
            for index, x in np.ndenumerate(flow):
                if x > 500:
                    flow[index] = 500

            # convert to polar
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # find out the maximum value across entire video
            # this will be used for normalization
            maxMagTmp = mag.max()
            if maxMagTmp > maxMag:
                maxMag = maxMagTmp

            # convert to HSV
            hsv[:, :, 0, indFlowMap] = ang * 180 / np.pi / 2
            hsv[:, :, 2, indFlowMap] = mag

            # Display the resulting frame
            tmp = hsv[:, :, :, indFlowMap].astype('B')  # convert to uint8
            frameProc = cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)
            imgDisplay = np.hstack((imgDisplay, frameProc))

            cv2.imshow('Previous, current frames and flow map', imgDisplay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prvs = next
            indFlowMap = indFlowMap + 1

    else:
        break

# When everything done, release the capture
cap.release()

# normalize the flow maps for each video
magNorm = np.divide(hsv[:, :, 2, :], maxMag)
magNorm = np.multiply(magNorm, 255)

# convert to uint8
hsv = hsv.astype('B')

# convert each frame from HSV to RGB and save them into a video file
for indFlowMap in range(numFlowMap):

    # conver from HSV to RGB for visualization
    frameProc = cv2.cvtColor(hsv[:, :, :, indFlowMap], cv2.COLOR_HSV2BGR)
    out.write(frameProc)

    # cv2.imshow('Previous, current frames and flow map', frameProc)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

out.release()

cv2.destroyAllWindows()

tEnd = time()
tElapsed = tEnd - tStart
print("time elapsed = %.2f seconds " % tElapsed)
