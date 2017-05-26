"""
******************************************************************************************************
 Project           Introduction to Signal and Image Processing - Group Project: Where is Waldo?
 Filename          KeyPointMatching.py

 Institution:      University of Bern

 Python            Python 3.6

 @author           Simon Scheurer, Yves Jegge
 @date             11.05.2016

 @status           Development

******************************************************************************************************
"""
# Import Package #
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from scipy import signal, misc
import cv2

# Import Modul #

"""
/*----------------------------------------------------------------------------------------------------
Method: FaceMatching()
------------------------------------------------------------------------------------------------------
This Method runs a face detection Algorithm.  Herby we need the Haarcascades-Algorithmen with the
generated Models will run. The Model were generated from 2300 picture
------------------------------------------------- -----------------------------------------------------
Input  Parameter:       Image as a input

Output Parameter:       Possible WaldoPositions
----------------------------------------------------------------------------------------------------*/
"""
def FaceMatching(image):

    # Settings for Line Matching #
    showDetectedFaces = True
    LoadModel = 1

    # Convert to Gray Image #
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Loading the CascadeClassifier-Model #
    if LoadModel == 1:
        face_cascade = cv2.CascadeClassifier('data/haarcascades/Cascade_WaldoHead_Stage23.xml')
    else:
        face_cascade = cv2.CascadeClassifier('data/haarcascades/Cascade_WaldoFace_Stage23.xml')

    # Detect Faces #
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(5,5), maxSize=(80,80))

    # Draw Rectangle #
    #if showDetectedFaces:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = image_gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        # Show Image
        #plt.imshow(image)
        #plt.show()

    # Return Computed image #
    return (image)
