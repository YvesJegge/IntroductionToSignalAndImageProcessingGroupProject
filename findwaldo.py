"""
******************************************************************************************************
 Project           Introduction to Signal and Image Processing - Group Project: Where is Waldo?
 Filename          findwaldo.py

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
import ColorMatching as cm
import TemplateMatching as tm
import ShapeMatching as sm
import FaceMatching as fm

"""
/*----------------------------------------------------------------------------------------------------
Method: find_waldo()
------------------------------------------------------------------------------------------------------
This Method takes an image and try to find the position of Waldo in the Image
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input

Output Parameter:       x,y coordinate of waldo
----------------------------------------------------------------------------------------------------*/
"""
def find_waldo(image):

    # Searching for Color that match  #
    image = cm.color_matching(image)

    # Searching for circles that match  #
    #image = sm.circle_matching(image)

    # Compute Template Matching
    template_matched_image_Hair = tm.hairfront_matching(image)
    template_matched_image_glasses = tm.eye_matching(image)

    # Searching for Shirts #
    matched_image_cap = sm.cap_matching(image)

    # Searching for Faces #
    matched_face = fm.FaceMatching(image)

    # Put all results together #
    matched_image = \
        1 * np.uint16(matched_image_cap) + \
        1 * np.uint16(template_matched_image_glasses) + \
        0.1 * np.uint16(template_matched_image_Hair) + \
        1 * np.uint16(matched_face)

    print("Max probability:" + str(np.max(matched_image)))


    # Blur dentisty
    matched_image = cv2.GaussianBlur(matched_image,(21,21),0)

    # Find Maximum Value of intensity Map #
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(matched_image)

    # Convert Coordinate Origin #
    x_coordinate = max_loc[0]
    y_coordinate = image.shape[0] - max_loc[1]

    # return position of Waldo #
    return x_coordinate, y_coordinate










