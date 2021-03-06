"""
******************************************************************************************************
 Project           Introduction to Signal and Image Processing - Group Project: Where is Waldo?
 Filename          KeyPointMatching.py

 Institution:      University of Bern

 Python            Python 3.6

 @author           Simon Scheurer, Yves Jegge
 @date             28.05.2016

 @status           Development

******************************************************************************************************
"""
# Import Package #
import numpy as np
import matplotlib.pyplot as plt
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

    if LoadModelNr == 1:
        # face_cascade.detectMultiScale(image_gray, scaleFactor = 1.9, minNeighbors = 6, minSize = (10, 10), maxSize = (30, 50) ==> 26 % #
        # Computed with "good" generated Faces, 20, 20 Pixel of Cascade #
        face_cascade = cv2.CascadeClassifier('data/haarcascades/Cascade_WaldoFace2_Stage16.xml')
    elif LoadModelNr == 2:
        #face_cascade.detectMultiScale(image_gray, 20, 20, minSize = (10, 10), maxSize = (30, 50) ) ==> 30 %
        # Computed with "good" generated Faces -numP 2000 -numNeg 1000 -numStages 10 (and 12, 16) -w 16 -h 24 #
        face_cascade = cv2.CascadeClassifier('data/haarcascades/Cascade_WaldoFace4_1_Stage12.xml')
    elif LoadModelNr == 3:
        # Computed with "good" generated Faces -numP 1000 -numNeg 2000 -numStages 10 (and 12, 16) -w 16 -h 24 #
        face_cascade = cv2.CascadeClassifier('data/haarcascades/Cascade_WaldoFace4_2_Stage10.xml')
    elif LoadModelNr == 4:
        # Computed with "good" generated Faces -numP 1000 -numNeg 1000 -numStages 10  (and 12 ,16) -w 16 -h 24 #
        face_cascade = cv2.CascadeClassifier('data/haarcascades/Cascade_WaldoFace4_3_Stage10.xml')
    elif LoadModelNr == 5:
        # Computed with "good" generated Faces and manualy faces -numP 2400 -numNeg 1200 -numStages 10  (and 12 ,16) -w 16 -h 24 #
        face_cascade = cv2.CascadeClassifier('data/haarcascades/Cascade_WaldoFace5_1_Stage10.xml')
    elif LoadModelNr == 6:
        # Computed with "good" generated Heads -numP 1800 - numNeg 900 - numStages 10 (and 12, 16) - w 20 - h 25 #
        face_cascade = cv2.CascadeClassifier('data/haarcascades/Cascade_WaldoHead2_1_Stage10.xml')

    # Detect Faces #
    faces = face_cascade.detectMultiScale(image_gray, 1.1 , 2, minSize = (18, 25), maxSize = (25, 35))

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
