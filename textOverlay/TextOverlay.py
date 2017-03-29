# This is a Python code using OpenCV to print the probabilities and predictions on videos
#
# I am working for a deadline. It's still messed up...
#
# Contact: Chih-Yao Ma at cyma@gatech.edu

import numpy as np
import cv2
import random 
import re

 
n = 3 # number of predictions per video
nVideo = 3754 # number of total videos

# Define the dimension of input frames
height = 240
width = 320
# what scale do you want to upscale the frames 
scale = 2

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (width*scale, height*scale))


# Read the list of predictions from txt file
with open('labels_rnn_20170328.txt') as rnn:
	linesRNN = rnn.readlines()
	numPred = np.size(linesRNN)
with open('labels_tcnn_20170328.txt') as tcnn:
	linesTCNN = tcnn.readlines()


# how many videos you want to use as demo
numDemoVideos = 10
indVideo = random.sample(range(1, nVideo), numDemoVideos)


# Start processing for each of the videos..
for number in indVideo: 

	idx = (number-1)*n

	# videoInfoRNN = linesRNN[number].split()	
	videoInfoRNN_1st = linesRNN[idx].split()
	videoInfoRNN_2nd = linesRNN[idx+1].split()
	videoInfoRNN_3rd = linesRNN[idx+2].split()

	# videoInfoTCNN = linesTCNN[number].split()
	videoInfoTCNN_1st = linesTCNN[idx].split()
	videoInfoTCNN_2nd = linesTCNN[idx+1].split()
	videoInfoTCNN_3rd = linesTCNN[idx+2].split()

	# Read the video file
	fileName = '/media/chih-yao/ssd-data/ucf101/video/' + videoInfoRNN_1st[0]
	cap = cv2.VideoCapture(fileName)

	# extract ground truth from file path
	groundTruth = videoInfoRNN_1st[0].split('/')[0]

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:

			# upscale the frames 
			frame = cv2.resize(frame, (0,0), fx = scale, fy = scale) 

			# transparent overlay
			overlay = frame.copy()
			cv2.rectangle(overlay, (30, 45), (600, 110), (20, 20, 20), -1)
			alpha = 0.5 # transparent ratio
			cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

			# get probabilities of predictions
			probRNN = map(float, [videoInfoRNN_1st[2], videoInfoRNN_2nd[2], videoInfoRNN_3rd[2]])
			probTCNN = map(float, [videoInfoTCNN_1st[2], videoInfoTCNN_2nd[2], videoInfoTCNN_3rd[2]])
			
			lengthRNN = [x * 100 for x in probRNN]
			lengthTCNN = [x * 100 for x in probTCNN]

			# print bars as the probabilities
			# probabilities for RNN 

			x_bar = 50
			y_bar = 55
			x_spacing = 300
			y_spacing = 20

			# probabilities for RNN
			frame = cv2.line(frame, (x_bar,y_bar), (x_bar + int(lengthRNN[0]), y_bar), (255,0,255), 5)
			frame = cv2.line(frame, (x_bar,y_bar+y_spacing), (x_bar + int(lengthRNN[1]), y_bar+y_spacing), (255,0,255), 5)
			frame = cv2.line(frame, (x_bar,y_bar+y_spacing*2), (x_bar + int(lengthRNN[2]), y_bar+y_spacing*2), (255,0,255), 5)

			# probabilities for TCNN
			frame = cv2.line(frame, (x_bar+x_spacing, y_bar), (x_bar+x_spacing + int(lengthTCNN[0]), y_bar), (255,0,255), 5)
			frame = cv2.line(frame, (x_bar+x_spacing, y_bar+y_spacing), (x_bar+x_spacing + int(lengthTCNN[1]), y_bar+y_spacing), (255,0,255), 5)
			frame = cv2.line(frame, (x_bar+x_spacing, y_bar+y_spacing*2), (x_bar+x_spacing + int(lengthTCNN[2]), y_bar+y_spacing*2), (255,0,255), 5)

			# Print predictions beside the probability bars
			# Print ground truth
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(frame, groundTruth, (225,20), font, 0.8, (0,255,0), 1, cv2.LINE_AA)

			x_pred = 175
			y_pred = 40

			# Print predictions from RNN
			cv2.putText(frame, 'TS-LSTM', (x_pred,y_pred), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
			cv2.putText(frame, videoInfoRNN_1st[1], (x_pred, y_pred+y_spacing), font, 0.5, (255,255,0), 1, cv2.LINE_AA)
			cv2.putText(frame, videoInfoRNN_2nd[1], (x_pred, y_pred+y_spacing*2), font, 0.5, (255,255,0), 1, cv2.LINE_AA)
			cv2.putText(frame, videoInfoRNN_3rd[1], (x_pred, y_pred+y_spacing*3), font, 0.5, (255,255,0), 1, cv2.LINE_AA)

			# Print predictions from TCNN
			cv2.putText(frame, 'Temporal-Inception', (x_pred+x_spacing, y_pred), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
			cv2.putText(frame, videoInfoTCNN_1st[1], (x_pred+x_spacing, y_pred+y_spacing), font, 0.5, (255,255,0), 1, cv2.LINE_AA)
			cv2.putText(frame, videoInfoTCNN_2nd[1], (x_pred+x_spacing, y_pred+y_spacing*2), font, 0.5, (255,255,0), 1, cv2.LINE_AA)
			cv2.putText(frame, videoInfoTCNN_3rd[1], (x_pred+x_spacing, y_pred+y_spacing*3), font, 0.5, (255,255,0), 1, cv2.LINE_AA)

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
