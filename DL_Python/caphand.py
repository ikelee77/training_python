import cv2
import numpy as np
import os, sys
from time import strftime, localtime
import random

cap = cv2.VideoCapture(0)
if not cap.isOpened(): sys.exit('camera error')

params = list()
params.append(cv2.IMWRITE_PNG_COMPRESSION)
params.append(8)
cwd = os.getcwd() 

while True:
	ret, frame = cap.read();
	if not ret: continue 
	rows, cols, channels = frame.shape
	width = cols
	height = rows
	length = min(width, height)
	pt = [60,60]
	if width < height: pt[1] += int((height - length) / 2)
	else: pt[0] += int((width - length) / 2)
	green = (0, 255, 0)  #BGR
	length -= 120
	cv2.rectangle(frame, (pt[0], pt[1]), (pt[0]+length, pt[1]+length), green, 4)
	cv2.imshow('view', frame)
	ch = cv2.waitKey(1) & 0xFF
	if ch == 27: break
	if ch == 32:
		mid_frame = frame[pt[1]:pt[1]+length, pt[0]:pt[0]+length]
		cut_frame = cv2.resize(mid_frame, (32, 32)) 
		cv2.imshow('cut', cut_frame)
		filename = cwd+'/'+strftime('%d%H%M%S', localtime())+str(random.randrange(1,10000))+'.png'
		cv2.imwrite(filename, cut_frame, params)