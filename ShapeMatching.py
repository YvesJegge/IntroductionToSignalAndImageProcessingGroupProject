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
    window_high = 50
    window_width = 50

    # Convert to hsv colorspace #
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Filter white #
    ws = (image_hsv[:, :, 1] < 130)
    wv = (image_hsv[:, :, 2] > 170)
    black_filtered = np.uint8(np.bitwise_and(ws, wv))

    # Filter black #
    #black_filtered = np.uint8((image_hsv[:, :, 2] < 160))

    # Separate colors of image #
    #red_filtered, white_filtered, pink_filtered, black_filtered = cm.separate_colors(image)

    # Finding Circles #
    circles = cv2.HoughCircles(black_filtered,cv2.HOUGH_GRADIENT, dp=1,minDist=4,param1=50,param2=13,minRadius=2,maxRadius=8)

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
    return black_filtered

"""
/*----------------------------------------------------------------------------------------------------
Method: eye_matching()
------------------------------------------------------------------------------------------------------
This Method search eyes
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input

Output Parameter:       New Image (Near Line there is the original picture, other black)
----------------------------------------------------------------------------------------------------*/
"""
def shirt_cap_matching(image):

    # Convert to hsv colorspace #
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Filter white #
    rh = np.bitwise_or((image_hsv[:, :, 0] < 5), (image_hsv[:, :, 0] > 172))
    rs = (image_hsv[:, :, 1] > 100)
    rv = (image_hsv[:, :, 2] > 140)
    red_filtered = np.uint8(np.bitwise_and(np.bitwise_and(rh, rs), rv))

    # Remove object with too small and too big size and wrong angle #
    filtered_img = cm.remove_image_objects(red_filtered, 10, 200, 1, 50, -1, 0.4)

    # Normalize array to Value 0-255 #
    filtered_img = cv2.dilate(filtered_img, np.ones((11,11)), iterations=1)
    filtered_img = cv2.normalize(filtered_img, filtered_img, 0, 255, cv2.NORM_MINMAX)
    filtered_img = cv2.blur(filtered_img,(11,11))


    return filtered_img