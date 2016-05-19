# Simple optical flow algorithm
# run for the whole UCF-101 dataset

# python version: 3.4.3
# OpenCV version: 3.1.0


# Contact:
# Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
# Chih-Yao Ma at <cyma@gatech.edu>
# Last update: 05/16/2016

import numpy as np
import cv2
import os
from scripts.flownet import FlowNet

# ----------------------------------------------
# --               Data paths                 --
# ----------------------------------------------

# dirDatabase = '/media/cmhung/MyDisk/CMHung_FS/Big_and_Data/PhDResearch/Code/Dataset/UCF-101/'
dirDatabase = '/home/chih-yao/Downloads/UCF-101/'

# ----------------------------------------------
# --                   Class                  --
# ----------------------------------------------
nameClass = os.listdir(dirDatabase)
nameClass.sort()
numClassTotal = len(nameClass)  # 101 classes

# Define the codec and create VideoWriter object
# fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D') # opencv 2.4
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv 3.0

# initialize the display window
cv2.namedWindow('Previous, current frames')
cv2.namedWindow('flow map')

for c in range(numClassTotal):  # c = 0 ~ 100
    c = 42
    dirClass = dirDatabase + nameClass[c] + '/'
    nameSubVideo = os.listdir(dirClass)
    numSubVideoTotal = len(nameSubVideo)  # videos

    outdir = dirDatabase + 'FlowMap' + '/' + nameClass[c] + '/'

    # create folder if not yet existed
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for sv in range(numSubVideoTotal):

        videoName = nameSubVideo[sv]
        videoPath = dirClass + videoName
        print('==> Loading the video: ' + videoName)

        cap = cv2.VideoCapture(videoPath)

        # information of the video
        Fr = round(1 / cap.get(2))  # frame rate
        Wd = int(cap.get(3))
        Ht = int(cap.get(4))
        nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get number of frames

        # output name
        nameParse = videoName.split(".")
        nameOutput = nameParse[0] + '_flow.' + nameParse[1]

        # check if the output file existed
        filename = outdir + nameOutput

        if not os.path.exists(filename):

            step = 3  # steps for computing optical flow
            numFlowMap = int(nFrame / step)

            # initialize video file with 1 FPS
            out = cv2.VideoWriter(filename, fourcc, 1, (Wd, Ht))

            # read the first frame
            ret, prvs = cap.read()
            prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)  # convert to gray scale
            # Get frame sizes
            height, width = prvs.shape

            # save in HSV (because of the optical flow algorithm we used)
            hsv = np.zeros((height, width, 3, numFlowMap)).astype('B')
            hsv[:, :, 1, :] = 255

            indFrame = 1
            indFlowMap = 0
            maxMag = 0

            while(cap.isOpened):
                # Capture frame-by-frame
                ret, next = cap.read()

                if ret is True:
                    indFrame = indFrame + 1

                    if ((indFrame - 1) % step) == 0 and indFlowMap < numFlowMap:

                        next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)  # convert to gray scale

                        imgDisplay = np.hstack((prvs, next))                     

                        flow = cv2.calcOpticalFlowFarneback(
                            prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        # show in RGB for visualization
                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        hsv[:, :, 0, indFlowMap] = ang * 180 / np.pi / 2
                        hsv[:, :, 2, indFlowMap] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                        frameProc = cv2.cvtColor(hsv[:, :, :, indFlowMap], cv2.COLOR_HSV2BGR)

                        # Display the resulting frame
                        # imgDisplay = np.hstack((imgDisplay, frameProc))

                        cv2.imshow('Previous, current frames', imgDisplay)
                        cv2.imshow('flow map', frameProc)

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
