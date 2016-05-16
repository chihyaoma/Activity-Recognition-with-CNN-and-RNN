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

#----------------------------------------------
#--               Data paths                 --
#----------------------------------------------

# dirDatabase = '/media/cmhung/MyDisk/CMHung_FS/Big_and_Data/PhDResearch/Code/Dataset/UCF-101/'

dirDatabase = '/home/chih-yao/Downloads/UCF-101/'

#----------------------------------------------
#--                   Class                  --
#----------------------------------------------
nameClass = os.listdir(dirDatabase)
numClassTotal = len(nameClass)  # 101 classes

# Define the codec and create VideoWriter object
# fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D') # opencv 2.4
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv 3.0

# initialize the display window
cv2.namedWindow('Previous, current frames and flow map')

for c in range(numClassTotal):  # c = 0 ~ 100
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
        # property identifier:
        # 1: ?; 2: s/frame; 3: width; 4: height; 6: ?; 7: ?
        Fr = round(1 / cap.get(2))
        Wd = int(cap.get(3))
        Ht = int(cap.get(4))

        # output name
        nameParse = videoName.split(".")
        nameOutput = nameParse[0] + '_flow.' + nameParse[1]

        # check if the output file existed
        filename = outdir + nameOutput

        if not os.path.exists(filename):
            
            out = cv2.VideoWriter(filename, fourcc, Fr, (Wd, Ht))

            # read the first frame
            ret, prvs = cap.read()
            # prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # convert to gray
            # scale

            # save in HSV (because of the optical flow algorithm we used)
            hsv = np.zeros_like(prvs)
            hsv[..., 1] = 255

            indFrame = 1

            while(cap.isOpened):
                # Capture frame-by-frame
                ret, next = cap.read()

                if (indFrame % 7) == 0:
                    if ret is True:

                        # Get frame sizes
                        height, width, channels = prvs.shape

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

                        for index, x in np.ndenumerate(flow):
                            if x > 100:
                                flow[index] = 0

                        # compute the optical flow from two adjacent frames
                        # next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                        # flow = cv2.calcOpticalFlowFarneback(
                        #   prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                        # show in RGB for visualization
                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        hsv[..., 0] = ang * 180 / np.pi / 2
                        hsv[..., 2] = cv2.normalize(
                            mag, None, 0, 255, cv2.NORM_MINMAX)
                        frameProc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                        out.write(frameProc)

                        # Display the resulting frame
                        imgDisplay = np.hstack((imgDisplay, frameProc))
                        # cv2.imshow(
                        #     'Previous, current frames and flow map for video: ' + nameParse[0], imgDisplay)

                        cv2.imshow('Previous, current frames and flow map', imgDisplay)
                        

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        prvs = next

                    else:
                        break

                indFrame = indFrame + 1

            # When everything done, release the capture
            cap.release()
            out.release()
        
cv2.destroyAllWindows()