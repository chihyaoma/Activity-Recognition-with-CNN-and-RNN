# FlowNet algorithm
# run for the whole UCF-101 dataset
# use the Middlebury color encoding method (C++ version)

# python version: 2.7.6 / 3.4.3
# OpenCV version: 3.1.0


# Contact:
# Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
# Chih-Yao Ma at <cyma@gatech.edu>

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

# steps for computing optical flow
step = 3

# ----------------------------------------------
# --               Data paths                 --
# ----------------------------------------------

dirDatabase = '/media/cmhung/MyDisk/CMHung_FS/Big_and_Data/PhDResearch/Code/Dataset/UCF-101/'
# dirDatabase = '/home/chih-yao/Downloads/UCF-101/'

# ----------------------------------------------
# --                   Class                  --
# ----------------------------------------------
inDir = dirDatabase + 'RGB/'
nameClass = os.listdir(inDir)
nameClass.sort()
numClassTotal = len(nameClass)  # 101 classes

# Define the codec and create VideoWriter object
# fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D') # opencv 2.4
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv 3.0

# initialize the display window
cv2.namedWindow('Previous, current frames and flow map')

idxClassAll = range(numClassTotal)

for c in idxClassAll[::-1]:  # c = 0 ~ 100 (start from the last one)
    tStart = time()  # calculate the time difference

    dirClass = inDir + nameClass[c] + '/'
    nameSubVideo = os.listdir(dirClass)
    nameSubVideo.sort()
    numSubVideoTotal = len(nameSubVideo)  # videos

    outdir = dirDatabase + 'FlowMap-M' + '/' + nameClass[c] + '/'

    # create folder if not yet existed
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    idxSubVideoAll = range(numSubVideoTotal)

    for sv in idxSubVideoAll[::-1]:  # also start from the last one

        videoName = nameSubVideo[sv]
        videoPath = dirClass + videoName
        print('==> Loading the video: ' + videoName)

        cap = cv2.VideoCapture(videoPath)

        # information of the video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        Wd = int(cap.get(3))
        Ht = int(cap.get(4))
        nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get number of frames

        # output name
        nameParse = videoName.split(".")
        nameOutput = nameParse[0] + '_flow.' + nameParse[1]

        # check if the output file existed
        pathVideoOut = outdir + nameOutput

        if not os.path.exists(pathVideoOut):
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

    # calculate the computation time
    tEnd = time()
    tElapsed = tEnd - tStart
    print("time elapsed of " + nameClass[c] + " = %.2f seconds " % tElapsed)

cv2.destroyAllWindows()
