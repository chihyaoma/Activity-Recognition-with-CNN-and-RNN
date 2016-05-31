# FlowNet algorithm
# for one video
# use the Middlebury color encoding method (python version)


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

#====== Middlebury Color Encoding Scheme ======#
# adapted from the color circle idea described at
# http://members.shaw.ca/quadibloc/other/colint.htm

RY = 15
YG = 6
GC = 4
CB = 11
BM = 13
MR = 6

ncols = RY + YG + GC + CB + BM + MR
colorwheel = np.zeros((ncols, 3))  # r, g, b

# RY
col = 0
tmp = 255 / RY * np.asarray(list(range(RY)))
colorwheel[0:RY, 0] = 255
colorwheel[0:RY, 1] = tmp.transpose()
col = col + RY

# YG
tmp = 255 / YG * np.asarray(list(range(YG)))
colorwheel[col:col + YG, 0] = 255 - tmp.transpose()
colorwheel[col:col + YG, 1] = 255
col = col + YG

# GC
tmp = 255 / GC * np.asarray(list(range(GC)))
colorwheel[col:col + GC, 1] = 255
colorwheel[col:col + GC, 2] = tmp.transpose()
col = col + GC

# CB
tmp = 255 / CB * np.asarray(list(range(CB)))
colorwheel[col:col + CB, 1] = 255 - tmp.transpose()
colorwheel[col:col + CB, 2] = 255
col = col + CB

# BM
tmp = 255 / BM * np.asarray(list(range(BM)))
colorwheel[col:col + BM, 2] = 255
colorwheel[col:col + BM, 0] = tmp.transpose()
col = col + BM

# MR
tmp = 255 / MR * np.asarray(list(range(MR)))
colorwheel[col:col + MR, 2] = 255 - tmp.transpose()
colorwheel[col:col + MR, 0] = 255

#==============================================#

# some common parameters about the color encoding
UNKNOWN_FLOW_THRESH = 1e9
UNKNOWN_FLOW = 1e10
ncols = np.shape(colorwheel)[0]

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
filename = 'FlowNet_out_M_Python.avi'

step = 5  # steps for computing optical flow

numFlowMap = int(nFrame / step)

# initialize video file with 1 FPS
out = cv2.VideoWriter(filename, fourcc, fps / step, (Wd, Ht))

# read the first frame
ret, prvs = cap.read()

# Get frame sizes
height, width, channels = prvs.shape

# # save in HSV (because of the optical flow algorithm we used)
# hsv = np.zeros((height, width, channels, numFlowMap))
# hsv[:, :, 1, :] = 255

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

            #====== Middlebury color encoding  ======#
            u = flow[:, :, 0]
            v = flow[:, :, 1]

            u[0, 0] = 0

            maxu = -999
            maxv = -999
            minu = 999
            minv = 999
            maxrad = -1

            # fix unknown flow
            idxUnknown = np.logical_or(np.nonzero(
                np.abs(u) > UNKNOWN_FLOW_THRESH), np.nonzero(np.abs(v) > UNKNOWN_FLOW_THRESH))

            u[idxUnknown] = 0
            v[idxUnknown] = 0

            maxu = max(maxu, np.max(u))
            minu = max(minu, np.min(u))

            maxv = max(maxv, np.max(v))
            minv = max(minv, np.min(v))

            rad = np.sqrt(np.square(u) + np.square(v))
            maxrad = max(maxrad, np.max(rad))

            u = u / (maxrad + np.spacing(1))
            v = v / (maxrad + np.spacing(1))

            # ----------------------------------------------------------------------
            # check if flow has nan value and assign them to zero
            nanIdx = np.logical_or(np.isnan(u), np.isnan(v))
            u[nanIdx] = 0
            v[nanIdx] = 0

            rad = np.sqrt(np.square(u) + np.square(v))

            a = np.arctan2(np.negative(v), np.negative(u)) / np.pi

            fk = (a + 1) / 2 * (ncols - 1) + 1  # -1~1 maped to 1~ncols
            k0 = np.floor(fk)  # 1, 2, ..., ncols

            k1 = k0 + 1

            # for index, x in np.ndenumerate(k1):
            #     if x == ncols+1:
            #         k1[index] = 1
            k1[k1 == ncols + 1] = 1

            f = fk - k0

            img = np.zeros((height, width, 3)).astype('B')
            col0 = np.zeros_like(k0)
            col1 = np.zeros_like(k1)

            for i in range(np.shape(colorwheel)[1]):

                tmp = colorwheel[..., i].flatten()

                for index, x in np.ndenumerate(k0):
                    col0[index] = tmp[x - 1] / 255
                for ind, y in np.ndenumerate(k1):
                    col1[ind] = tmp[y - 1] / 255

                col = np.multiply((1 - f), col0) + np.multiply(f, col1)

                idx = np.nonzero(rad <= 1)
                # increase saturation with radius
                col[idx] = 1 - rad[idx] * (1 - col[idx])

                nonidx = np.zeros_like(col)
                nonidx[idx] = 1
                idx = np.nonzero(nonidx != 1)
                col[idx] = col[idx] * 0.75  # out of range

                img[:, :, 2 - i] = (np.floor((255 * col) * (1 - nanIdx))
                                    )  # to uint8
                img[idxUnknown, 2 - i] = 0
            #========================================#

            #====== HSV color encoding w/ the whole video normalization ======#
            # # prune the flow value if it's weirdly large
            # for index, x in np.ndenumerate(flow):
            #     if x > 500:
            #         flow[index] = 500

            # # convert to polar
            # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # # find out the maximum value across entire video
            # # this will be used for normalization
            # maxMagTmp = mag.max()
            # if maxMagTmp > maxMag:
            #     maxMag = maxMagTmp

            # # convert to HSV
            # hsv[:, :, 0, indFlowMap] = ang * 180 / np.pi / 2
            # hsv[:, :, 2, indFlowMap] = mag

            # # Display the resulting frame
            # tmp = hsv[:, :, :, indFlowMap].astype('B')  # convert to uint8
            # frameProc = cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)
            #=================================================================#

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

#====== HSV color encoding w/ the whole video normalization ======#
# # normalize the flow maps for each video
# magNorm = np.divide(hsv[:, :, 2, :], maxMag)
# magNorm = np.multiply(magNorm, 255)

# # convert to uint8
# hsv = hsv.astype('B')

# # convert each frame from HSV to RGB and save them into a video file
# for indFlowMap in range(numFlowMap):

#     # conver from HSV to RGB for visualization
#     frameProc = cv2.cvtColor(hsv[:, :, :, indFlowMap], cv2.COLOR_HSV2BGR)
#     out.write(frameProc)

#     # cv2.imshow('Previous, current frames and flow map', frameProc)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break
#=================================================================#

out.release()

cv2.destroyAllWindows()

tEnd = time()
tElapsed = tEnd - tStart
print("time elapsed = %.2f seconds " % tElapsed)
