# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
import serial

######## Communicating to ARDUINO ########
######## Initialise SERIAL ########
ser = serial.Serial('COM4', 9800, timeout=1)

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

vs = cv2.VideoCapture(0)
# allow the camera or video file to warm up
time.sleep(2.0)

while True:
	# grab the current frame
	frame = vs.read()
	frame = frame[1]

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
    
	dimensions = frame.shape
	width = frame.shape[1]
    
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		cx=center[0]
		cy=center[1]
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
			cv2.line(frame,(cx,0),(cx,720),(25,56,4),1)
			cv2.line(frame,(0,cy),(1280,cy),(25,56,4),1)
			cv2.circle(frame, (cx, cy), 5, (48, 39, 210), 2)
            
			if cx <= ((width)//2 -10) and cx>0:
				print ("Turn Left!")
				ser.write(b'l')
			
			elif (cx < ((width//2)+ 10) and cx > ((width//2) - 10)):
				print ("On Track!")
				ser.write(b'f')

			elif cx >= ((width//2 +10 )):
				print ("Turn Right")
				ser.write(b'r')

			elif cx==0:
				print ("I don't see the line")

			cv2.line(frame,((width//2)- 10,0),((width//2)- 30,dimensions[0]),(255,0,125),2)    # Parallel to x-axis
			cv2.line(frame,((width//2)+ 10,0),((width//2)+ 30,dimensions[0]),(255,0,125),2)    # Parallel to y-axis

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

ser.close()
vs.release()
cv2.destroyAllWindows()
