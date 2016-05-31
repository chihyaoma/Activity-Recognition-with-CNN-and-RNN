# FlowNet algorithm
# for one video
# use the Middlebury color encoding method (C++ version)


# python version: 2.7.6 / 3.4.3
# OpenCV version: 3.1.0

# Contact:
# Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
# Chih-Yao Ma at <cyma@gatech.edu>
#
# Last update: 05/31/2016

import numpy as np
import cv2
import os
from scripts.flownet import FlowNet
from time import time

# import C++ function
import sys
sys.path.insert(1, './colorflow_Python_C++/build/lib.linux-x86_64-2.7/')
import ColorFlow

os.environ['GLOG_minloglevel'] = '3' # suppress the output

#==============================================#

# read the video file
nameVideoIn = 'v_Archery_g01_c06'
# nameVideoIn = 'v_Basketball_g01_c01'
# nameVideoIn = 'v_Basketball_g03_c06'
# nameVideoIn = 'v_TaiChi_g01_c01'
# nameVideoIn = 'v_HorseRiding_g01_c06'
# nameVideoIn = 'v_PlayingGuitar_g01_c01'
# nameVideoIn = 'v_HighJump_g01_c05'
# nameVideoIn = 'v_LongJump_g01_c06'
# nameVideoIn = 'v_ParallelBars_g02_c02'



cap = cv2.VideoCapture(nameVideoIn + '.avi')

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

# output name
step = 5  # steps for computing optical flow
nameVideoOut = nameVideoIn + '_M_C++_' + str(step)

dirOut = 'VideoOutputs'
# create folder if not yet existed
if not os.path.exists(dirOut):
    os.makedirs(dirOut)

pathVideoOut = dirOut + '/' + nameVideoOut + '.avi'


numFlowMap = int(nFrame / step)

# initialize video file with 1 FPS
out = cv2.VideoWriter(pathVideoOut, fourcc, fps / step, (Wd, Ht))

# read the first frame
ret, prvs = cap.read()

# Get frame sizes
height, width, channels = prvs.shape

indFrame = 1
indFlowMap = 0
maxMag = 0

tStart = time()  # calculate the time difference

#==============================================#

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
            # the FlowNet will save a .flo file (input is a frame?)
            FlowNet.run(prvs)

            # read the .flo file
            nameFlow = 'flownetc-pred-0000000.flo'
            # temporary output image in the Middlebury color style
            nameOutTemp = 'outTemp.ppm'

            # convert the flow file to the color image file
            ColorFlow.flow2color(nameOutTemp, nameFlow)

            # read the temporary output image (.ppm)
            img = cv2.imread(nameOutTemp)

            imgDisplay = np.hstack((imgDisplay, img))
            out.write(img)

            cv2.imshow('Previous, current frames and flow map', imgDisplay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prvs = next
            indFlowMap = indFlowMap + 1

    else:
        break

# When everything done, release the capture
cap.release()

out.release()

cv2.destroyAllWindows()

# calculate the computation time
tEnd = time()
tElapsed = tEnd - tStart
print("time elapsed = %.2f seconds " % tElapsed)
