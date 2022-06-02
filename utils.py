import numpy as np
import time
import cv2 
import pyttsx3
import keyboard
import requests
from pathlib import Path
from tqdm import tqdm
import speech_recognition as sr
import pytesseract

def init():
	engine = pyttsx3.init()
	engine.setProperty("rate", 178)
	r = sr.Recognizer()
	return engine, r

engine, r = init()

def say(out_say):
    engine.say(out_say)
    engine.runAndWait()

def download_file(filename, url):
	chunkSize = 1024
	r = requests.get(url, stream=True)

	with open(filename, 'wb') as f:
		pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))

		for chunk in r.iter_content(chunk_size=chunkSize):
			if chunk: # filter out keep-alive new chunks
				pbar.update(len(chunk))

				f.write(chunk)

	return filename

def take_command():
	r = sr.Recognizer()

	with sr.Microphone() as source:

		r.pause_threshold = 0.5
		print("Listening...")
		audio = r.listen(source)

	try:
		query = r.recognize_google(audio, language='en')
		print(f"You said: {query}\n")
	except Exception as e:
		print(e)
		return ""

	return query

# def runnnnnnnn():
	pass
	# while True:
	# 	if keyboard.is_pressed("q"):
	# 		break
	# 	if "what do you see" in take_command().lower():
	# # determine only the *output* layer names that we need from YOLO
	# 		ln = net.getLayerNames()
	# 		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# 		# initialize
	# 		cap = cv2.VideoCapture(0)
	# 		frame_count = 0
	# 		start = time.time()
	# 		first = True
	# 		frames = []

	# 		while True:
	# 			frame_count += 1

	# 			if keyboard.is_pressed("n"):
	# 				break
				
	# 			# Capture frame-by-frame
	# 			ret, frame = cap.read()
	# 			frame = cv2.flip(frame, 1)
	# 			frames.append(frame)

	# 			if not ret: continue # if no frame, go to next iteration

	# 			cv2.imshow('Video Feed', cv2.flip(frame, 1))
	# 			key = cv2.waitKey(1)

	# 			if frame_count % 100: # runs predictor once every 100 frames
	# 				continue  # guard, if its not 100th frame, skip the loop

	# 			end = time.time()

	# 			(H, W) = frame.shape[:2]
				
	# 			blob = cv2.dnn.blobFromImage(
	# 				frame, 
	# 				1 / 255.0, 
	# 				(416, 416),
	# 				swapRB = True, 
	# 				crop = False
	# 			)

	# 			net.setInput(blob)
	# 			layerOutputs = net.forward(ln)
				
	# 			boxes = []
	# 			confidences = []
	# 			classIDs = []
	# 			centers = []

	# 			for output in layerOutputs:
	# 				for detection in output:
	# 					scores = detection[5:]
	# 					classID = np.argmax(scores)
	# 					confidence = scores[classID]

	# 					if confidence > 0.5:
	# 						box = detection[0:4] * np.array([W, H, W, H])
	# 						(centerX, centerY, width, height) = box.astype("int")
							
	# 						x = int(centerX - (width / 2))
	# 						y = int(centerY - (height / 2))
							
	# 						boxes.append([x, y, int(width), int(height)])
	# 						confidences.append(float(confidence))
	# 						classIDs.append(classID)
	# 						centers.append((centerX, centerY))

	# 			idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

	# 			texts = []

	# 			if len(idxs) <= 0:  # no idxs so dont run the next code
	# 				continue

	# 			for i in idxs.flatten():
	# 				centerX, centerY = centers[i]

	# 				if centerX <= W/3:
	# 					W_pos = "left "
	# 				elif centerX <= (W/3 * 2):
	# 					W_pos = "center "
	# 				else:
	# 					W_pos = "right "

	# 				if centerY <= H/3:
	# 					H_pos = "top "
	# 				elif centerY <= (H/3 * 2):
	# 					H_pos = "mid "
	# 				else:
	# 					H_pos = "bottom "

	# 				texts.append(H_pos + W_pos + LABELS[classIDs[i]])


	# 			if texts:
	# 				description = ', '.join(texts)
	# 				say(description)

	# 		cap.release()
	# 		cv2.destroyAllWindows()
	# 	if "read" in take_command().lower():
	# 		say("Position your camera to the piece of text")
	# 		time.sleep(2.5)
	# 		cap = cv2.VideoCapture(0)
	# 		ret, frame = cap.read()
	# 		img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# 		read(img_rgb)

def detect(image, verbose=0):
	LABELS = open("coco.names").read().strip().split("\n")


	if not Path('yolov3.weights').is_file():
		download_file("yolov3.weights", "https://pjreddie.com/media/files/yolov3.weights")
	if verbose: print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
	if verbose: print("Done!")

	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()] # changed i[0] - 1 to i - 1
	image = cv2.flip(image, 1)
	(H, W) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(
				image, 
				1 / 255.0, 
				(416, 416),
				swapRB = True, 
				crop = False
			)
	net.setInput(blob)
	layerOutputs = net.forward(ln)
				
	boxes = []
	confidences = []
	classIDs = []
	centers = []
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > 0.5:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
							
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
							
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				centers.append((centerX, centerY))

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
	texts = []

	if len(idxs) <= 0:  # no idxs so dont run the next code
		say("Nothing Found. Try Again")
		return "Nothing found"

	for i in idxs.flatten():
		centerX, centerY = centers[i]

		if centerX <= W/3:
			W_pos = "left "
		elif centerX <= (W/3 * 2):
			W_pos = "center "
		else:
			W_pos = "right "

		if centerY <= H/3:
			H_pos = "top-"
		elif centerY <= (H/3 * 2):
			H_pos = "mid-"
		else:
			H_pos = "bottom-"

		texts.append(H_pos + W_pos + LABELS[classIDs[i]])


	if texts:
		description = ', '.join(texts)
		print(description)
		say(description)

def read(img): 
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	text = pytesseract.image_to_string(img_rgb)
	print(text)
	say(text)

# img = cv2.imread("C:/Users/abc/Downloads/PP_78.jpg")
# read(img)
# detect(img)