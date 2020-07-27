import numpy as np
import imutils
import pickle
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os
import face_recognition
#python <filename>.py --image <imagepath> --recognizer <output/recognizer.pickle> --le output/le.pickle 

# construct the argument parser and parse the arguments
def align(image1,shape_predictor):
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor and the face aligner
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor)
	fa = FaceAligner(predictor, desiredFaceWidth=256)

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(image1)
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 2)
	faceAligned = fa.align(image, gray, rects[0])
	return faceAligned


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")

ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-n", "--name", required=True,
	help="name entered while marking attendance")
args = vars(ap.parse_args()) 

# load our serialized face detector from disk
detector = dlib.get_frontal_face_detector()
# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())



#image = cv2.imread(args["image"])
image= align(args["image"], 'shape_predictor_68_face_landmarks.dat')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes
# corresponding to each face in the input image
boxes = face_recognition.face_locations(rgb,
    model='hog')

# compute the facial embedding for the face
encodings = face_recognition.face_encodings(rgb, boxes)
print(len(encodings))
for encoding in encodings:
    encoding= encoding.reshape(1,-1)
    preds = recognizer.predict_proba(encoding)[0]
    j = np.argmax(preds)
    proba = preds[j]
    name = le.classes_[j]
    print(name)
    print(proba*100)
    if(name==args["name"]) and (proba*100>50):
    	print("True")
    else:
    	print("False")

