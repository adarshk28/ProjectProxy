# ProjectProxy
 Covid Project on Facial Recognition Software
 
 Made by Adarsh Kumar and Tulip Pandey
 
 Description - 
 A software that detects faces, classifies the input images as real or fake and finally matches the real faces with the entries in the dataset, returning the best match. The recognition  of faces has been done using an SVM model and the liveness  detection ( fake/ real) classification has been done using a CNN model, trained on our own dataset.
 
 
 Packages and Tools used - 
 1. OpenCV 
 2. dlib
 3. imutils
 4. pickle
 5. face_recognition library
 6. tensorflow
 7. keras
 8. numpy
 9. matplotlib 
 10. os
 
 How to run -
 1. Step 0 - run encode_faces.py -- this will encode all the images in the dataset and store the encoding in the folder 'pickles/encodings.pickle'
 2. Step 1 - run svmmodel.py -- this creates an svm model and fits our dataset to the model for face recognition
 3. Step 2 - run train_liveness.py -- this fits our dataset to a CNN model for liveness detection 
 4. Step 3 - run projectproxy.py -- this records a small video clip and returns the name of the best match from the dataset 
 
 
 
 
