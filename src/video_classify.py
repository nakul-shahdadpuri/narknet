#imports

import cv2
import numpy as np
import time
import sys

#Load Yolo
print("Loading Model.....")
net = cv2.dnn.readNet('../weights/yolov3.weights','../cfg/yolov3.cfg')
classes = []

with open('../dataset/coco.names','r') as f:
	classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

print("Model Loaded")

#image loading
file = sys.argv[1]
cap = cv2.VideoCapture("../tests/" + file)
f = open('../results/dump.csv', 'a+') 
time_now = time.time()
frame_id = 0
font = cv2.FONT_HERSHEY_PLAIN

while True:
	_,frame = cap.read()
	try:
		frame = cv2.resize(frame,(800,800))
	except Exception:
		print("ERROR")
		exit(1)
	frame_id = frame_id + 1
	height, width, channels = frame.shape
	#detecting object

	blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0), True, crop = False)

	# for b in blob:
	# 	for n,img_blob in enumerate(b):
	# 		cv2.imshow(str(n),img_blob)

	net.setInput(blob)
	outs = net.forward(output_layers)

	#outs to image
	boxes = []
	confidences = []
	class_ids = []

	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.3:
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h/2)
				boxes.append([x,y,w,h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	#supressing similar boxes
	indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

	objects_detected = len(boxes)
	for i in range(objects_detected):
		if i in indexes:
			x,y,w,h = boxes[i]
			label = classes[class_ids[i]] + " " + str(int(confidences[i]*100)) + "%"
			s = str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + ',' + classes[class_id] + ',' + str(int(confidences[i]*100)) + '\n'
			f.write(s)
			print(x,y,w,h,label)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.putText(frame,label,(x,y+30),font,1,(0,0,255),2)

	time_passed = time.time() - time_now
	fps = frame_id/time_passed
	cv2.putText(frame,"fps: "+ str(fps),(10,30),font, 2,(255,255,255),1)
	cv2.imshow('Video',frame)
	key = cv2.waitKey(1)
	if key == 27:
		break
cap.release()
cv2.destroyAllWindows()
f.close()