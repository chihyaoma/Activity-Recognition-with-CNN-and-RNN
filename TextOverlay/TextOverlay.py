# This is a Python code to print the probabilities of predictions on videos
#
#
#
# Contact: Chih-Yao Ma at cyma@gatech.edu
# 05/03/2016

import numpy as np
import cv2

# Read the list of 
with open('filename') as f:
    lines = f.readlines()

cap = cv2.VideoCapture('SampleVideo.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))


while(cap.isOpened()):
	ret, frame = cap.read()
	if ret==True:
	
		# get probabilities of predictions
		prob = [0.75, 0.14, 0.11]
		length = [x * 100 for x in prob]
		

		# print a bar as the probability 
		frame = cv2.line(frame, (50,20), (50 + int(length[0]), 20), (255,0,255), 10)
		frame = cv2.line(frame, (50,40), (50 + int(length[1]), 40), (255,0,255), 10)
		frame = cv2.line(frame, (50,60), (50 + int(length[2]), 60), (255,0,255), 10)
		
		# Print predictions beside the probability bars
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame,'Class #1',(150,30), font, 0.5,(255,0,0),1,cv2.LINE_AA)
		cv2.putText(frame,'Class #2',(150,50), font, 0.5,(255,0,0),1,cv2.LINE_AA)
		cv2.putText(frame,'Class #3',(150,70), font, 0.5,(255,0,0),1,cv2.LINE_AA)

		# write the processed frame
		out.write(frame)

		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()