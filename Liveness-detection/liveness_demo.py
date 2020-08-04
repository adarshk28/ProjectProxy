# import the necessary packages
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import face_recognition
import dlib

def align(image,shape_predictor):
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor and the face aligner
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor)
	fa = FaceAligner(predictor, desiredFaceWidth=256)

	# load the input image, resize it, and convert it to grayscale
	#image = cv2.imread(image1)
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 2)
	if len(rects)==0:
		return None 
	faceAligned = fa.align(image, gray, rects[0])
	return faceAligned

def takeSecond(elem):
	return elem[1]


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-l", "--le_liveness", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-le", "--le", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
args = vars(ap.parse_args())

#python liveness_demo.py --model liveness.model --le_liveness le.pickle --le output/le.pickle --recognizer output/recognizer.pickle
model = load_model(args["model"])
le_liveness = pickle.loads(open(args["le_liveness"], "rb").read())
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
LABEL=''
t=0
rcntr=0
# loop over the frames from the video stream
dict1=[]
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	# grab the frame dimensions and convert it to a blob

	detection = face_recognition.face_locations(frame,
	model='hog')
	face=frame

	for i in range(len(detection)):
		
		(top,right,bottom,left)  = detection[0]

		face = frame[top:bottom,left:right]
		
		face1 = cv2.resize(face, (32, 32))
		face=face1
		face1 = face.astype("float") / 255.0
		#face1=face
		face1 = img_to_array(face1)
		face1 = np.expand_dims(face1, axis=0)
		# pass the face ROI through the trained liveness detector
		# model to determine if the face is "real" or "fake"
		preds = model.predict(face1)[0]
		j = np.argmax(preds)
		label = le_liveness.classes_[j]
		LABEL=label
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
	if t==3:
		break
	if(LABEL=='Real'):
		recognizer = pickle.loads(open(args["recognizer"], "rb").read())
		le = pickle.loads(open(args["le"], "rb").read())



		#image = cv2.imread(args["image"])
		image= align(face, 'shape_predictor_68_face_landmarks.dat')
		if image is None:
			continue
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input image
		boxes = face_recognition.face_locations(rgb,
		    model='hog')

		# compute the facial embedding for the face
		encodings = face_recognition.face_encodings(rgb, boxes)

		#for encoding in encodings:
		encoding= encodings[0].reshape(1,-1)
		preds = recognizer.predict_proba(encoding)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]
		dict1.append((name,proba))
		rcntr+=1
		   # print(name)
		    #print(proba*100)
	t+=1

dict1.sort(reverse=True,key=takeSecond)
if dict1[0][1]>0.50 and rcntr==t:
	print(dict1[0][0])
else:
	print("Try Again :/")
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# realcounter=0 -> counts the names printed; if less than original count, try again
# empty dictionary -> name is the key and match is the value-> sort the dictionary as per values and return the name with maximum proba
#