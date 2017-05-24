"""
******************************************************************************************************
 Project           Introduction to Signal and Image Processing - Group Project: Where is Waldo?
 Filename          ColorMatching.py

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

"""
/*----------------------------------------------------------------------------------------------------
Method: color_matching()
------------------------------------------------------------------------------------------------------
This Method takes the original image and find via color matching waldo
------------------------------------------------------------------------------------------------------
Input  Parameter:       original_image

Output Parameter:       Give probability map back with the same size of original image (0....1)
----------------------------------------------------------------------------------------------------*/
"""
def color_matching(image):
    # Settings for color Matching #
    show_color = False

    # Blurring image #
    image_blurred = cv2.GaussianBlur(image,(5,3),0)

    # Convert to hsv colorspace #
    image_hsv = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2HSV)

    # Filter red #
    rh = np.bitwise_or((image_hsv[:, :, 0] < 5), (image_hsv[:, :, 0] > 172))
    rs = (image_hsv[:, :, 1] > 100)
    rv = (image_hsv[:, :, 2] > 140)
    red_filtered = np.uint8(np.bitwise_and(np.bitwise_and(rh, rs), rv))

    # Filter white #
    wh = np.bitwise_or((image_hsv[:, :, 0] < 65), (image_hsv[:, :, 0] > 165))
    ws = (image_hsv[:, :, 1] < 90)
    wv = (image_hsv[:, :, 2] > 170)
    white_filtered = np.uint8(np.bitwise_and(np.bitwise_and(wh, ws), wv))

    # Filter pink #
    ph = np.bitwise_or((image_hsv[:, :, 0] < 10), (image_hsv[:, :, 0] > 172))
    ps = (image_hsv[:, :, 1] < 90)
    pv = (image_hsv[:, :, 2] > 140)
    pink_filtered = np.uint8(np.bitwise_and(np.bitwise_and(ph, ps), pv))

    # Filter black #
    black_filtered = np.uint8((image_hsv[:, :, 2] < 98))

    # Kernels #
    kernel_noise = np.ones((2,2))
    kernel_small = np.ones((4,3))
    kernel_big = np.ones((30,10))

    # Remove small objects #
    black_filtered = cv2.morphologyEx(black_filtered, cv2.MORPH_OPEN, kernel_noise)

    # Remove object with too small and too big size #
    red_filtered = remove_image_objects(red_filtered, 10, 200)
    white_filtered = remove_image_objects(white_filtered, 1, 300)
    pink_filtered = remove_image_objects(pink_filtered, 5, 200)
    black_filtered = remove_image_objects(black_filtered, 12, 5200)

    # Dilate filters (make objects bigger) #
    red_filtered = cv2.dilate(red_filtered, kernel_small, iterations=1)
    white_filtered = cv2.dilate(white_filtered, kernel_small, iterations=1)
    pink_filtered = cv2.dilate(pink_filtered, kernel_small, iterations=1)
    black_filtered = cv2.dilate(black_filtered, kernel_small, iterations=1)

    # Find overlaps #
    strips_filtered = np.multiply(red_filtered, white_filtered)
    hair_hut_filtered = np.multiply(red_filtered, black_filtered)
    hair_face_filtered = np.multiply(pink_filtered, black_filtered)
    strips_face_filtered = np.multiply(pink_filtered, strips_filtered)

    # Dilate filters (make objects bigger) #
    strips_filtered = cv2.dilate(strips_filtered, kernel_big, iterations=1)
    hair_hut_filtered = cv2.dilate(hair_hut_filtered, kernel_big, iterations=1)
    hair_face_filtered = cv2.dilate(hair_face_filtered, kernel_big, iterations=1)
    strips_face_filtered = cv2.dilate(strips_face_filtered, kernel_big, iterations=1)


    # Find overlaps #
    color_filtered = strips_filtered + hair_hut_filtered + hair_face_filtered + strips_face_filtered

    # Dilate filters (make objects bigger) #
    color_filtered = cv2.dilate(color_filtered, kernel_big, iterations=3)

    if show_color:

        plt.figure(200)
        plt.subplot(2,2,1)
        plt.imshow(red_filtered)
        plt.subplot(2,2,2)
        plt.imshow(white_filtered)
        plt.subplot(2,2,3)
        plt.imshow(pink_filtered)
        plt.subplot(2,2,4)
        plt.imshow(black_filtered)

        plt.figure(201)
        plt.subplot(2,2,1)
        plt.imshow(strips_filtered)
        plt.subplot(2,2,2)
        plt.imshow(hair_hut_filtered)
        plt.subplot(2,2,3)
        plt.imshow(hair_face_filtered)
        plt.subplot(2,2,4)
        plt.imshow(strips_face_filtered)

    # Cut out only matched areas #
    filtered_img = cv2.bitwise_and(image, image, mask = np.uint8(color_filtered > 3))

    # Normalize array to Value 0-255 #
    # cv2.normalize(color_filtered, color_filtered, 0, 255, cv2.NORM_MINMAX)

    return filtered_img

"""
/*----------------------------------------------------------------------------------------------------
Method: remove_image_objects()
------------------------------------------------------------------------------------------------------
This Method deletes object which are too small or too big
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input, min size of objects, max size of objects

Output Parameter:       image without object which are too small or too big
----------------------------------------------------------------------------------------------------*/
"""
def remove_image_objects(img, min_size, max_size):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    #for every component in the image, you keep it only if it's above min_size
    area_size = sizes[output-1]
    img2 = np.uint8(np.multiply((area_size >= min_size), (area_size <= max_size)))

    return img2