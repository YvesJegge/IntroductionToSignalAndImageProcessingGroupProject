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
    LoadModelNr = 2
    show_detected_Faces_in_image = False
    show_filtered_image = False

    # Convert to Gray Image #
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Loading the CascadeClassifier-Model #
    if LoadModelNr  == 1:
        face_cascade = cv2.CascadeClassifier('data/haarcascades/Cascade_WaldoHead1_Stage23.xml')
    elif LoadModelNr == 2:
        face_cascade = cv2.CascadeClassifier('data/haarcascades/Cascade_WaldoFace1_Stage16.xml')

    # Detect Faces #
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10,10), maxSize=(50,50))

    # Filtering image (detected Face become Value 255 )  #
    filtered_img = np.zeros(image_gray.shape).astype(np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(filtered_img , (x, y), (x + w, y + h), 255, -1)

    # Normalize array to Value 0-255 #
    filtered_img = cv2.dilate(filtered_img, np.ones((11, 11)), iterations=1)
    filtered_img = cv2.normalize(filtered_img, filtered_img, 0, 255, cv2.NORM_MINMAX)
    filtered_img = cv2.blur(filtered_img, (11, 11))

    # Draw Rectangle in Original Picture #
    if show_detected_Faces_in_image:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plt.figure(500)
        plt.imshow(image)
        plt.title("Detected Faces")

    # Show filtered image  #
    if show_filtered_image:
        plt.figure(501)
        plt.imshow(filtered_img, cmap='gray')
        plt.title("Denisity Map of Detected Faces")

    # Show both images at the same time #
    if show_detected_Faces_in_image or show_filtered_image:
        plt.show()

    # Return Density image #
    return (filtered_img)

if __name__ == "__main__":

    img = plt.imread("data/images_1/20.jpg").astype(np.uint8)
    FaceMatching(img)