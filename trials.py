# import pytesseract
# import cv2
# import pyttsx3

# engine = pyttsx3.init("sapi5")
# engine.setProperty("rate", 150)

# def say(out_say):
    # print(out_say)
    # engine.say(out_say)
    # engine.save_to_file(out_Say, 'test.mp3') # Use this if you can't use pyttsx3 in Streamlit.
    # engine.runAndWait()
# def ocr(img_rgb): say(pytesseract.image_to_string(img_rgb))

# ocr("C:/Users/abc/Downloads/IMG_20220515_214901_647.jpg")


# def run(image):
	# LABELS = open("coco.names").read().strip().split("\n")


	# if not Path('yolov3.weights').is_file():
	# 	download_file("yolov3.weights", "https://pjreddie.com/media/files/yolov3.weights")

	# # load our YOLO object detector trained on COCO dataset (80 classes)
	# print("[INFO] loading YOLO from disk...")
	# net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
	# print("Done!")

	# ln = net.getLayerNames()
	# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	# image = cv2.flip(image, 1)
	# blob = cv2.dnn.blobFromImage(
	# 			frame, 
	# 			1 / 255.0, 
	# 			(416, 416),
	# 			swapRB = True, 
	# 			crop = False
	# 		)
	# net.setInput(blob)
	# layerOutputs = net.forward(ln)
				
	# boxes = []
	# confidences = []
	# classIDs = []
	# centers = []
	# for output in layerOutputs:
	# 	for detection in output:
	# 		scores = detection[5:]
	# 		classID = np.argmax(scores)
	# 		confidence = scores[classID]

	# 		if confidence > 0.5:
	# 			box = detection[0:4] * np.array([W, H, W, H])
	# 			(centerX, centerY, width, height) = box.astype("int")
							
	# 			x = int(centerX - (width / 2))
	# 			y = int(centerY - (height / 2))
							
	# 			boxes.append([x, y, int(width), int(height)])
	# 			confidences.append(float(confidence))
	# 			classIDs.append(classID)
	# 			centers.append((centerX, centerY))

	# idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
	# texts = []

	# if len(idxs) <= 0:  # no idxs so dont run the next code
	# 	continue

	# for i in idxs.flatten():
	# 	centerX, centerY = centers[i]

	# 	if centerX <= W/3:
	# 		W_pos = "left "
	# 	elif centerX <= (W/3 * 2):
	# 		W_pos = "center "
	# 	else:
	# 		W_pos = "right "

	# 	if centerY <= H/3:
	# 		H_pos = "top "
	# 	elif centerY <= (H/3 * 2):
	# 		H_pos = "mid "
	# 	else:
	# 		H_pos = "bottom "

	# 	texts.append(H_pos + W_pos + LABELS[classIDs[i]])


	# if texts:
	# 	description = ', '.join(texts)
	# 	say(description)


# import tensorflow as tf
# import tensorflow_hub as hub
# import pydub 
# import numpy as np

# model = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2-960h/1")
# # For using this model, it's important to set `jit_compile=True` on GPUs/CPUs
# # as some operations in this model (i.e. group-convolutions) are unsupported without it
# model = tf.function(model, jit_compile=True)
# model(audio)

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import numpy as np
import time
import cv2
import utils

st.title("Seer")
class VideoProcessor:
	def __init__(self):
		self.command = "Blah"
	def run(self, img):
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		if "what do you see" in self.command.lower():
			time.sleep(2.5)
			utils.detect(img)
		elif "read" in self.command.lower():
			utils.say("Position the camera over the piece of text")
			time.sleep(2.5)
			utils.ocr(img_rgb)
		else:
			pass
	def recv(self, frame):
		img = frame.to_ndarray(format="bgr24")
		self.run(img)
		return frame


ctx = webrtc_streamer(key="Seer", video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False})
ctx.command = utils.take_command()
