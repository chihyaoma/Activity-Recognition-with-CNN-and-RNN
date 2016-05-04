# This is a Python code to print the probabilities and predictions on videos
#
#
#
# Contact: Chih-Yao Ma at cyma@gatech.edu
# 05/03/2016

import numpy as np
import cv2
import random 
import re


# Define the dimension of input frames
height = 240
width = 320
# what scale do you want to upscale the frames 
scale = 2

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (width*scale, height*scale))


# Read the list of predictions from txt file
with open('labels_rnn.txt') as f:
	lines = f.readlines()
	numPred = np.size(lines)

# how many videos you want to use as demo
numDemoVideos = 3
indVideo = random.sample(range(1, numPred), numDemoVideos)


# Start processing for each of the videos..
for number in indVideo: 

	# videoInfo = lines[indVideo[0]].split()
	videoInfo = lines[number].split()

	# Read the video file
	fileName = '/home/chih-yao/Downloads/UCF-101/' + videoInfo[0]
	cap = cv2.VideoCapture(fileName)

	# extract ground truth from file path
	idx = videoInfo[0].find('/')
	groundTruth = videoInfo[0][:idx]

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:

			# upscale the frames 
			frame = cv2.resize(frame, (0,0), fx = scale, fy = scale) 
		
			# get probabilities of predictions
			prob = [0.75, 0.14, 0.11]
			length = [x * 100 for x in prob]
			

			# print a bar as the probability 
			frame = cv2.line(frame, (50,40), (50 + int(length[0]), 40), (255,0,255), 5)
			frame = cv2.line(frame, (50,60), (50 + int(length[1]), 60), (255,0,255), 5)
			frame = cv2.line(frame, (50,80), (50 + int(length[2]), 80), (255,0,255), 5)
			
			# Print predictions beside the probability bars
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(frame, groundTruth, (150,20), font, 0.5,(0,255,0), 1, cv2.LINE_AA)
			cv2.putText(frame, videoInfo[1], (150,40), font, 0.5,(255,255,0), 1, cv2.LINE_AA)
			cv2.putText(frame, 'Class #2', (150,60), font, 0.5,(255,255,0), 1, cv2.LINE_AA)
			cv2.putText(frame, 'Class #3', (150,80), font, 0.5,(255,255,0), 1, cv2.LINE_AA)

			# write the processed frame
			out.write(frame)

			cv2.imshow('frame',frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

	# Release everything if job is finished
	cap.release()
	cv2.destroyAllWindows()

# release output 
out.release()
