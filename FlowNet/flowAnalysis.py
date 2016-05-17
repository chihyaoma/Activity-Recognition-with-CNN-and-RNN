# When the movement of the objects in the video is not distinct to be
# captured by optical flow algorithm, training this "noisy" flow map
# against the ground truth labeling is risky. In this code, we would
# like to iterate through all the generated flow videos, and filter
# out the noisy flow map.
#
#
# Contact: Chih-Yao Ma at cyma@gatech.edu
# Last update: 05/17/2016

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

# cap = cv2.VideoCapture('v_HandStandPushups_g01_c04_flow.avi')
cap = cv2.VideoCapture('v_HandStandPushups_g12_c06_flow.avi')


# information of the video
# property identifier:
# 1: ?; 2: s/frame; 3: width; 4: height; 6: ?; 7: ?
Fr = round(1 / cap.get(2))
Wd = int(cap.get(3))
Ht = int(cap.get(4))

# Define the codec and create VideoWriter object
# fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D') # opencv 2.4
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv 3.0
out = cv2.VideoWriter('out_flow.avi', fourcc, Fr, (Wd, Ht))

indFrame = 1

def close_event():
    plt.close() #timer calls this function after 3 seconds and closes the window 


while(cap.isOpened):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        print('--------------------------------------')
        print('Frame # ', indFrame)

        # convert back to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # extract the channels and flat them
        channel_0 = hsv[..., 0].flatten()
        channel_1 = hsv[..., 1].flatten()
        channel_2 = hsv[..., 2].flatten()

        # out.write(frame)
        # Display the resulting frame
        cv2.imshow('Processed frame', frame)

        # plot histogram for each channel 
        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
        
        ax0.hist(channel_0, 20, normed=1, histtype='bar', facecolor='r', alpha=0.75)
        ax0.set_title('Channel #0')
        ax1.hist(channel_1, 20, normed=1, histtype='bar', facecolor='g', alpha=0.75)
        ax1.set_title('Channel #1')
        ax2.hist(channel_2, 20, normed=1, histtype='bar', facecolor='b', alpha=0.75)
        ax2.set_title('Channel #2')

        # plot the figure for a short time
        plt.tight_layout()

        timer = fig.canvas.new_timer(interval = 4000) #creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(close_event)
        timer.start()
        plt.show()
        # fname = 'histogramFrame_' + str(indFrame)
        # plt.savefig(fname)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break      

    else:
        break
    indFrame = indFrame + 1

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
