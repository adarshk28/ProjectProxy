# import the necessary packages
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import face_recognition
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
	help="path to label encoder")
args = vars(ap.parse_args())

#python liveness_demo.py --model liveness.model --le le.pickle 

model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	# grab the frame dimensions and convert it to a blob

	detection = face_recognition.face_locations(frame,
	model='hog')

	for i in range(len(detection)):
		
		(top,right,bottom,left)  = detection[0]

		face = frame[top:bottom,left:right]

		face = cv2.resize(face, (32, 32))
		face = face.astype("float") / 255.0
		face = img_to_array(face)
		face = np.expand_dims(face, axis=0)
		# pass the face ROI through the trained liveness detector
		# model to determine if the face is "real" or "fake"
		preds = model.predict(face)[0]
		j = np.argmax(preds)
		label = le.classes_[j]
		# draw the label and bounding box on the frame
		label = "{}: {:.4f}".format(label, preds[j])
		cv2.putText(frame, label, (left, top - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 0, 255), 2)

            # show the output frame and wait for a key press
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()