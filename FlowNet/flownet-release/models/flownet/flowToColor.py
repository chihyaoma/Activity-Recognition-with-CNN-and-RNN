# Convert flow map to color image
#
#
#
#
# Contact:
# Chih-Yao Ma at <cyma@gatech.edu>

import numpy as np
from math import floor
import cv2

# class flowToColor:

# def makeColorWheel():

# color encoding scheme
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


# -------------------------------------------------------------------------

# def flowToColor(flow)

height = 720
width = 1280

# read the .flo file
fileName = 'flownetc-pred-0000000.flo'
flowMapSize = np.fromfile(fileName, np.float32, count=1)
if flowMapSize != 202021.25:
    print 'Dimension incorrect. Invalid .flo file'
else:
    data = np.fromfile(fileName, np.float32,
                       count=2 * width * height)

flow = np.resize(data, (height, width, 2))

UNKNOWN_FLOW_THRESH = 1e9
UNKNOWN_FLOW = 1e10

u = flow[:,:,0]
v = flow[:,:,1]

u[0,0] = 0

maxu = -999
maxv = -999
minu = 999
minv = 999
maxrad = -1

# fix unknown flow
idxUnknown = np.logical_or(np.nonzero(np.abs(u)>UNKNOWN_FLOW_THRESH), np.nonzero(np.abs(v)>UNKNOWN_FLOW_THRESH))

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

ncols = np.shape(colorwheel)[0]

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
        col0[index] = tmp[x-1] / 255
    for ind, y in np.ndenumerate(k1):
        col1[ind] = tmp[y-1] / 255

    col = np.multiply((1 - f), col0) + np.multiply(f, col1)

    idx = np.nonzero(rad <= 1)
    col[idx] = 1 - rad[idx] * (1 - col[idx])  # increase saturation with radius

    nonidx = np.zeros_like(col)
    nonidx[idx] = 1
    idx = np.nonzero(nonidx != 1)
    col[idx] = col[idx] * 0.75  # out of range


    img[:, :, 2-i] = (np.floor((255 * col) * (1 - nanIdx))
                    )  # to uint8
    img[idxUnknown,2-i] = 0

print(img.shape)
# cv2.namedWindow('flow map in RGB')
# cv2.imshow('flow map in RGB', img)
cv2.imwrite('test.png', img)