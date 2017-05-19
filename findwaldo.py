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
from scipy import signal
# Import Modul #
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

    # Compute Template Matching
    template_matched_image = template_matching(image, "data/templates/WaldoSmall.jpeg")

    # Only for Testing Intensity Map #
    display_denisty_map(image, template_matched_image)

    # Find Maximum Value of intensity Map #
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(template_matched_image)

    # Convert Coordinate Origin #
    # (maybe not needed) #
    x_coordinate = max_loc[0]
    y_coordinate = image.shape[0] - max_loc[1]

    # return position of Waldo #
    return x_coordinate, y_coordinate

"""
/*----------------------------------------------------------------------------------------------------
Method: display_denisty_map()
------------------------------------------------------------------------------------------------------
This Method takes the original image and the denisty image and print int out
------------------------------------------------------------------------------------------------------
Input  Parameter:       original_image, denisty_image

Output Parameter:       None
----------------------------------------------------------------------------------------------------*/
"""
def display_denisty_map(original_image, denisty_image):
    # Plot Original Image  #
    plt.figure(100)
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis('on')
    plt.title('Original Map')
    # Plot Density Map #
    plt.subplot(1, 2, 2)
    plt.imshow(denisty_image)

    plt.axis('on')
    plt.title('Intensity Map')
    plt.show

"""
/*----------------------------------------------------------------------------------------------------
Method: template_matching()
------------------------------------------------------------------------------------------------------
This Method runs a template matching algorithm throws the image
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input, template path

Output Parameter:       Density image that is generated from the template matching
----------------------------------------------------------------------------------------------------*/
"""
def template_matching(image, template_path):

    # Read in Template Picture #
    template = cv2.imread(template_path)

    # Compute Template Matching #
    template_matched_image = cv2.matchTemplate(image, template, method=cv2.TM_CCOEFF)

    # Return template matched picture #
    return template_matched_image




