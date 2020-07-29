#USAGE python gather_examples.py --input dataset_videos --output dataset_snaps --skip 20 
# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import face_recognition
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to output directory of cropped faces")
#removed detector and confidence
ap.add_argument("-s", "--skip", type=int, default=16,
	help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())



# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0
# loop over frames from the video file stream
while True:
	# grab the frame from the file
	(grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# increment the total number of frames read thus far
	read += 1

	# check to see if we should process this frame
	if read % args["skip"] != 0:
		continue

	# pass the frame through the network and obtain the detections and
	# predictions

	detection = face_recognition.face_locations(frame,
	model='hog')
	

	(top,right,bottom,left)  = detection
	face = blob[bottom:top, left:right]
	p = os.path.sep.join([args["output"],
			"{}.png".format(saved)])
	cv2.imwrite(p, face)
	saved += 1

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()





