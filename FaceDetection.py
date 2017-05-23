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
Method: face_detection()
------------------------------------------------------------------------------------------------------
This Method runs a face detection Algorithm.  Herby we need the Haarcascades-Algorithmen with the
according Models
------------------------------------------------- -----------------------------------------------------
Input  Parameter:       Image as a input

Output Parameter:       Density image that is generated from the template matching
----------------------------------------------------------------------------------------------------*/
"""
def face_detection(image):

    # Settings for Line Matching #
    showDetectedFaces = True

    # Convert to Gray Image #
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Loading the CascadeClassifier-Model #
    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

    # Detect Faces #
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    # Draw Rectangle #
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = image_gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Show Image with detected Faces #
    if showDetectedFaces:
        plt.imshow(image)
        plt.show()






"""
-----------------------------------------------------------------------------------------------------
Main Programm
-----------------------------------------------------------------------------------------------------
This File wil control / test the group project Where is Waldo
-----------------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":

    image = plt.imread('data/TestImagesHaarcascades/Faces_3.jpg').astype(np.uint8)

    face_detection(image)

