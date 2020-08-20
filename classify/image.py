#imports

import sys
import cv2
import numpy as np



def load_model(PATH_TO_WEIGHTS='narknet/weights/yolov3.weights',PATH_TO_CFG='narknet/cfg/yolov3.cfg',PATH_TO_NAMES='narknet/dataset/coco.names'):
	print("Loading Model.....")
	net = cv2.dnn.readNet(PATH_TO_WEIGHTS,PATH_TO_CFG)
	classes = []

	with open(PATH_TO_NAMES,'r') as f:
		classes = [line.strip() for line in f.readlines()]

	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	print("Model Loaded")
	return net,classes,output_layers,layer_names


def predict(PATH_TO_OBJECT,net,classes,output_layers,layer_names):
#image loading
	try:
		img = cv2.imread(PATH_TO_OBJECT)
		height, width, channels = img.shape
	except :
		print("ERROR: " +PATH_TO_OBJECT + " DOES NOT EXIST")
		exit(0)

	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

	net.setInput(blob)
	outs = net.forward(output_layers)

	#outs to image
	boxes = []
	CONFIDENCES = []
	class_ids = []

	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h/2)  	
				boxes.append([x, y, w, h])
				CONFIDENCES.append(float(confidence))
				class_ids.append(class_id)

	#supressing similar boxes
	indexes = cv2.dnn.NMSBoxes(boxes, CONFIDENCES, 0.5, 0.4)

	objects_detected = len(boxes)
	font = cv2.FONT_HERSHEY_PLAIN
	meows = []
	for i in range(objects_detected):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = classes[class_ids[i]] + " " + str(int(CONFIDENCES[i]*100)) + "%"
			meow = str(x)+','+str(y)+','+str(w)+','+str(h)+','+label
			meows.append(meow)
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
			cv2.putText(img, label, (x, y+30), font, 1, (0, 0, 255), 2)

	return img,meows
