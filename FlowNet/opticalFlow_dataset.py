# Simple optical flow algorithm 
# run for the whole UCF-101 dataset

# python version: 3.4.3
# OpenCV version: 3.1.0


# Contact: Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
# Last update: 05/13/2016

import numpy as np
import cv2
import os

#----------------------------------------------
#-- 			  Data paths 			     --
#----------------------------------------------
dirDatabase = '/media/cmhung/MyDisk/CMHung_FS/Big_and_Data/PhDResearch/Code/Dataset/UCF-101/'

#----------------------------------------------
#-- 			      Class		        	 --
#----------------------------------------------
nameClass = os.listdir(dirDatabase)
numClassTotal = len(nameClass)  # 101 classes

# Define the codec and create VideoWriter object
# fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D') # opencv 2.4
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv 3.0

for c in range(numClassTotal):  # c = 0 ~ 100
	dirClass = dirDatabase + nameClass[c] + '/'
	nameSubVideo = os.listdir(dirClass)
	numSubVideoTotal = len(nameSubVideo)  # videos

	for sv in range(numSubVideoTotal):

		videoName = nameSubVideo[sv]
		videoPath = dirClass + videoName
		print('==> Loading the video: ' + videoName)

		cap = cv2.VideoCapture(videoPath)

		# information of the video
		# property identifier:
		# 1: ?; 2: s/frame; 3: width; 4: height; 6: ?; 7: ?
		Fr = round(1 / cap.get(2))
		# Fr = 25
		Wd = int(cap.get(3))
		Ht = int(cap.get(4))

		# print(Fr)
		# print(Wd)
		# print(Ht)

		# output name
		nameParse = videoName.split(".")
		nameOutput = nameParse[0] + '_flow.' + nameParse[1]
		out = cv2.VideoWriter(dirClass + nameOutput, fourcc, Fr, (Wd, Ht))

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

				ch = 0xFF & cv2.waitKey(Fr)
				if ch == 27:
					break
				prvs = next

			else:
				break

		# When everything done, release the capture
		cap.release()
		out.release()
		cv2.destroyAllWindows()
