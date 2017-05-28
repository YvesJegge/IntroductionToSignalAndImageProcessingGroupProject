"""
******************************************************************************************************
 Project           Introduction to Signal and Image Processing - Group Project: Where is Waldo?
 Filename          ShapeMatching.py

 Institution:      University of Bern

 Python            Python 3.6

 @author           Simon Scheurer, Yves Jegge
 @date             24.05.2016

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

"""
/*----------------------------------------------------------------------------------------------------
Method: circle_matching()
------------------------------------------------------------------------------------------------------
This Method search for circles in the image (for searching or for glasses, hood, nose, eyes).
This algorithm need first a preprocessingof the picture
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input

Output Parameter:       New Image (Near circle there is the original picture, other black)
----------------------------------------------------------------------------------------------------*/
"""
def circle_matching(image):

    # Settings for circle Matching #
    show_circle_in_image = False
    show_filtered_image = False
    window_high = 70
    window_width = 70

    # Convert to Gray Image #
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Finding Circles #
    circles = cv2.HoughCircles(image_gray,cv2.HOUGH_GRADIENT, dp=1,minDist=4,param1=50,param2=13,minRadius=2,maxRadius=8)

    # Filtering image (near a circle coping parts from original image)  #
    filtered_image = np.zeros(image.shape).astype(np.uint8)
    if circles is not None:
        for i in circles[0, :]:
            # Compute start and stop Value of Window #
            x_start = i[0] - (window_width/2)
            x_stop = i[0] + (window_width/2)
            y_start =i[1] - (window_high/2)
            y_stop = i[1] + (window_high/2)
            # Check for Array Boundary mismatch #
            x_start = 0 if x_start < 0 else x_start
            x_stop = image.shape[1] if x_stop > image.shape[1] else x_stop
            y_start = 0 if y_start < 0 else y_start
            y_stop = image.shape[0] if y_stop > image.shape[1] else y_stop
            # Convert to int for Array Indexing #
            x_start = int(x_start)
            x_stop = int(x_stop)
            y_start = int(y_start)
            y_stop = int(y_stop)
            # Copying part of the Image
            filtered_image[y_start:y_stop, x_start:x_stop, :] = image[y_start:y_stop, x_start:x_stop, :]

    # Showing the Circles in the original image #
    if show_circle_in_image:
        if circles is not None:
            for i in circles[0, :]:
                # Draw the outer circle #
                cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 0), 2)
                # Draw the center of the circle #
                cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 1)
        plt.figure(300)
        plt.imshow(image)
        plt.title("Before Circle Filtering")

    # Show filtered image  #
    if show_filtered_image:
        plt.figure(301)
        plt.imshow(filtered_image)
        plt.title("After Circle Filtering")

    # Show both images at the same time
    if show_circle_in_image or show_filtered_image:
        plt.show()

    # Return circle map #
    return filtered_image


"""
/*----------------------------------------------------------------------------------------------------
Method: cap_matching()
------------------------------------------------------------------------------------------------------
This Method search caps
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input

Output Parameter:       New Image (Near Line there is the original picture, other black)
----------------------------------------------------------------------------------------------------*/
"""


def cap_matching(image):
    # Convert to hsv colorspace #
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Filter red #
    rh = np.bitwise_or((image_hsv[:, :, 0] < 6), (image_hsv[:, :, 0] > 165))
    rs = (image_hsv[:, :, 1] > 50)
    rv = (image_hsv[:, :, 2] > 140)
    red_filtered = np.uint8(np.bitwise_and(np.bitwise_and(rh, rs), rv))

    # Filter black #
    black_filtered = np.uint8((image_hsv[:, :, 2] < 98))

    # Remove small objects
    red_filtered = cm.remove_image_objects(red_filtered, 8, 150, 0.8, 5, -1.5, -0.4)
    black_filtered = cv2.erode(black_filtered, np.ones((2, 3)), iterations=1)
    black_filtered = cm.remove_image_objects(black_filtered, 2, 1000, 0.25, 4, -2, 2)

    # Make hair and cap bigger
    red_filtered = cv2.dilate(red_filtered, np.ones((4, 6)), iterations=1)
    black_filtered = cv2.dilate(black_filtered, np.ones((4, 6)), iterations=1)

    # Find overlaps of hair and cap #
    filtered_img = np.bitwise_and(red_filtered, black_filtered)

    # Remove object with too small and too big size and wrong angle #
    filtered_img = cm.remove_image_objects(filtered_img, 1, 150, 0.8, 5, -2.5, 2.5)

    # Normalize array to Value 0-255 #
    filtered_img = cv2.dilate(filtered_img, np.ones((30, 15)), iterations=1)
    filtered_img = cv2.normalize(filtered_img, filtered_img, 0, 255, cv2.NORM_MINMAX)
    filtered_img = cv2.blur(filtered_img, (30, 15))

    return filtered_img



"""
/*----------------------------------------------------------------------------------------------------
Method: shirt_matching()
------------------------------------------------------------------------------------------------------
This Method search shirts
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input

Output Parameter:       New Image (Near Line there is the original picture, other black)
----------------------------------------------------------------------------------------------------*/
"""


def shirt_matching(image):
    # Convert to hsv colorspace #
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Filter red #
    rh = np.bitwise_or((image_hsv[:, :, 0] < 6), (image_hsv[:, :, 0] > 165))
    rs = (image_hsv[:, :, 1] > 50)
    rv = (image_hsv[:, :, 2] > 140)
    red_filtered = np.uint8(np.bitwise_and(np.bitwise_and(rh, rs), rv))

    # Filter black #
    black_filtered = np.uint8((image_hsv[:, :, 2] < 98))

    # Remove small objects
    #red_filtered = cm.remove_image_objects(red_filtered, 20, 150, 2, 100, -0.5, 0.5)
    red_filtered= cv2.erode(red_filtered, np.ones((2, 3)), iterations=1)
    red_filtered = cm.remove_image_objects(red_filtered, 10, 150, 0.5, 5, -1, 1)



    # Normalize array to Value 0-255 #
    #filtered_img = cv2.dilate(filtered_img, np.ones((30, 15)), iterations=1)
    #filtered_img = cv2.normalize(filtered_img, filtered_img, 0, 255, cv2.NORM_MINMAX)
    #filtered_img = cv2.blur(filtered_img, (30, 15))

    return red_filtered