# This is a Python code using OpenCV to print the probabilities and predictions on videos
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
with open('labels_rnn.txt') as rnn:
	linesRNN = rnn.readlines()
	numPred = np.size(linesRNN)
with open('labels_tcnn.txt') as tcnn:
	linesTCNN = tcnn.readlines()


# how many videos you want to use as demo
numDemoVideos = 3
indVideo = random.sample(range(1, numPred), numDemoVideos)


# Start processing for each of the videos..
for number in indVideo: 

	
	videoInfoRNN = linesRNN[number].split()
	videoInfoTCNN = linesTCNN[number].split()

	# Read the video file
	fileName = '/home/chih-yao/Downloads/UCF-101/' + videoInfoRNN[0]
	cap = cv2.VideoCapture(fileName)

	# extract ground truth from file path
	groundTruth = videoInfoRNN[0].split('/')[0]

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:

			# upscale the frames 
			frame = cv2.resize(frame, (0,0), fx = scale, fy = scale) 
		
			# get probabilities of predictions
			prob = [0.75, 0.14, 0.11]
			length = [x * 100 for x in prob]
			

			# print bars as the probabilities
			# probabilities for RNN 

			x_bar = 50
			y_bar = 60
			x_spacing = 300
			y_spacing = 20

			# probabilities for RNN
			frame = cv2.line(frame, (x_bar,y_bar), (x_bar + int(length[0]), y_bar), (255,0,255), 5)
			frame = cv2.line(frame, (x_bar,y_bar+y_spacing), (x_bar + int(length[1]), y_bar+y_spacing), (255,0,255), 5)
			frame = cv2.line(frame, (x_bar,100), (x_bar + int(length[2]), 100), (255,0,255), 5)

			# probabilities for TCNN
			frame = cv2.line(frame, (x_bar+x_spacing,y_bar), (x_bar+x_spacing + int(length[0]), y_bar), (255,0,255), 5)
			frame = cv2.line(frame, (x_bar+x_spacing,y_bar+y_spacing), (x_bar+x_spacing + int(length[1]), y_bar+y_spacing), (255,0,255), 5)
			frame = cv2.line(frame, (x_bar+x_spacing,y_bar+y_spacing*2), (x_bar+x_spacing + int(length[2]), y_bar+y_spacing*2), (255,0,255), 5)

			# Print predictions beside the probability bars
			# Print ground truth
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(frame, groundTruth, (225,20), font, 0.8,(0,255,0), 1, cv2.LINE_AA)

			x_pred = 150
			y_pred = 40

			# Print predictions from RNN
			cv2.putText(frame, 'RNN', (x_pred,y_pred), font, 0.5,(0,0,255), 1, cv2.LINE_AA)
			cv2.putText(frame, videoInfoRNN[1], (x_pred,y_pred+y_spacing), font, 0.5,(255,255,0), 1, cv2.LINE_AA)
			cv2.putText(frame, 'Class #2', (x_pred,y_pred+y_spacing*2), font, 0.5,(255,255,0), 1, cv2.LINE_AA)
			cv2.putText(frame, 'Class #3', (x_pred,y_pred+y_spacing*3), font, 0.5,(255,255,0), 1, cv2.LINE_AA)

			# Print predictions from TCNN
			cv2.putText(frame, 'TCNN', (x_pred+x_spacing,y_pred), font, 0.5,(0,0,255), 1, cv2.LINE_AA)
			cv2.putText(frame, videoInfoTCNN[1], (x_pred+x_spacing,y_pred+y_spacing), font, 0.5,(255,255,0), 1, cv2.LINE_AA)
			cv2.putText(frame, 'Class #2', (x_pred+x_spacing,y_pred+y_spacing*2), font, 0.5,(255,255,0), 1, cv2.LINE_AA)
			cv2.putText(frame, 'Class #3', (x_pred+x_spacing,y_pred+y_spacing*3), font, 0.5,(255,255,0), 1, cv2.LINE_AA)

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
