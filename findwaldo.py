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

    #color_matching(image)
    color_matched_image = color_matching(image)

    # Compute Template Matching
    template_matched_image = template_matching(image, "data/templates/WaldoFace.jpg")

    # Compute keypoint_detection #
    # (Maybe better than Template Matching, however not yet implemented) #
    #template_matched_image = keypoint_detection(image, "data/templates/WaldoSmall.jpeg")

    # Put all results together #
    matched_image = np.multiply(color_matched_image, template_matched_image)

    # Only for Testing Intensity Map #
    display_denisty_map(image, matched_image)

    # Find Maximum Value of intensity Map #
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(matched_image)

    # Convert Coordinate Origin #
    # (maybe not needed) #
    x_coordinate = max_loc[0]
    y_coordinate = image.shape[0] - max_loc[1]

    # return position of Waldo #
    return x_coordinate, y_coordinate

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

    # Blurring image #
    image = cv2.GaussianBlur(image,(9,3),0)

    # Convert to hsv colorspace #
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Filter red #
    rh = np.bitwise_or((image_hsv[:, :, 0] < 6), (image_hsv[:, :, 0] > 154))
    rs = (image_hsv[:, :, 1] > 100)
    rv = (image_hsv[:, :, 2] > 135)
    red_filtered = np.uint8(np.bitwise_and(np.bitwise_and(rh, rs), rv))

    # Filter white #
    wh = np.bitwise_or((image_hsv[:, :, 0] < 70), (image_hsv[:, :, 0] > 160))
    ws = (image_hsv[:, :, 1] < 100)
    wv = (image_hsv[:, :, 2] > 170)
    white_filtered = np.uint8(np.bitwise_and(np.bitwise_and(wh, ws), wv))

    # Filter pink #
    ph = np.bitwise_or((image_hsv[:, :, 0] < 25), (image_hsv[:, :, 0] > 170))
    ps = (image_hsv[:, :, 1] < 130)
    pv = (image_hsv[:, :, 2] > 120)
    pink_filtered = np.uint8(np.bitwise_and(np.bitwise_and(ph, ps), pv))

    # Filter black #
    black_filtered = np.uint8((image_hsv[:, :, 2] < 70))

    # Kernels #
    kernel_noise_small = np.ones((2,1))
    kernel_noise_big = np.ones((3,3))
    kernel_small = np.ones((3,1))
    kernel_big = np.ones((9,6))

    # Opening filters (remove noise) #
    red_filtered = cv2.morphologyEx(red_filtered, cv2.MORPH_OPEN, kernel_noise_small)
    pink_filtered = cv2.morphologyEx(pink_filtered, cv2.MORPH_OPEN, kernel_noise_big)
    black_filtered = cv2.morphologyEx(black_filtered, cv2.MORPH_OPEN, kernel_noise_big)

    # Dilate filters (make objects bigger) #
    red_filtered = cv2.dilate(red_filtered, kernel_small, iterations=1)
    white_filtered = cv2.dilate(white_filtered, kernel_small, iterations=2)
    pink_filtered = cv2.dilate(pink_filtered, kernel_small, iterations=5)
    black_filtered = cv2.dilate(black_filtered, kernel_small, iterations=1)

    # Find overlaps #
    strips_filtered = np.multiply(red_filtered, white_filtered)
    hair_hut_filtered = np.multiply(red_filtered, black_filtered)
    hair_face_filtered = np.multiply(pink_filtered, black_filtered)

    # Dilate filters (make objects bigger) #
    strips_filtered = cv2.dilate(strips_filtered, kernel_big, iterations=1)
    hair_hut_filtered = cv2.dilate(hair_hut_filtered, kernel_big, iterations=1)
    hair_face_filtered = cv2.dilate(hair_face_filtered, kernel_big, iterations=1)

    # Find overlaps #
    strips_hut_hair_filtered = np.multiply(strips_filtered, hair_hut_filtered)
    hut_hair_face_filtered = np.multiply(hair_hut_filtered, hair_face_filtered)

    # Dilate filters (make objects bigger) #
    strips_hut_hair_filtered = cv2.dilate(strips_hut_hair_filtered, kernel_big, iterations=5)
    hut_hair_face_filtered = cv2.dilate(hut_hair_face_filtered, kernel_big, iterations=5)

    # Find overlaps #
    color_filtered = np.multiply(strips_hut_hair_filtered, hut_hair_face_filtered)

    #plt.figure(2)
    #plt.subplot(2,2,1)
    #plt.imshow(image[50:120, 1910:1940])
    #plt.subplot(2,2,2)
    #plt.imshow(strips_filtered[50:120, 1910:1940])
    #plt.subplot(2,2,3)
    #plt.imshow(hair_hut_filtered[50:120, 1910:1940])
    #plt.subplot(2,2,4)
    #plt.imshow(hair_face_filtered[50:120, 1910:1940])

    #plt.figure(3)
    #plt.subplot(2,2,1)
    #plt.imshow(red_filtered[50:120, 1910:1940])
    #plt.subplot(2,2,2)
    #plt.imshow(white_filtered[50:120, 1910:1940])
    #plt.subplot(2,2,3)
    #plt.imshow(pink_filtered[50:120, 1910:1940])
    #plt.subplot(2,2,4)
    #plt.imshow(black_filtered[50:120, 1910:1940])


    #plt.figure(4)
    #plt.subplot(2,2,1)
    #plt.imshow(strips_hut_hair_filtered[50:120, 1910:1940])
    #plt.subplot(2,2,2)
    #plt.imshow(hut_hair_face_filtered[50:120, 1910:1940])
    #plt.subplot(2,2,3)
    #plt.imshow(image[50:120, 1910:1940])
    #plt.subplot(2,2,4)
    #plt.imshow(color_filtered[50:120, 1910:1940])

    # Normalize array to Value 0-255 #
    cv2.normalize(color_filtered, color_filtered, 0, 255, cv2.NORM_MINMAX)

    return color_filtered





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
    plt.imshow(denisty_image, cmap='gray')

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

    #-- Set Settings for template Matching-- #
    gray_picture = True
    canny_detection = True
    blur_filter = False

    # Read in Template Picture #
    template = plt.imread(template_path)

    # Filtering the Image #
    if blur_filter:
        image = cv2.medianBlur(image, 5)
        template =cv2.medianBlur(template, 5)

    # Convert Image and template to Gray-Scale #
    if gray_picture:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    # Edge detection for Template #
    if canny_detection:
        # Compute for Canny edge dedection for image #
        max_magnitude_image = np.median(image)
        thr_high_image = 0.3 * max_magnitude_image
        thr_low_image = thr_high_image / 2
        image = cv2.Canny(image, threshold1=thr_low_image, threshold2=thr_high_image)
        # Compute for Canny edge dedection for template #
        max_magnitude_template = np.median(template)
        thr_high_template = 0.2 * max_magnitude_template
        thr_low_template = thr_high_template / 2
        template = cv2.Canny(template, threshold1=thr_low_template, threshold2=thr_high_template)


    # Initialize used variable #
    best_template_match = image
    (template_hight, template_width) = template.shape[:2]
    best_max_val = None

    # -- Loop over the scales of the image -- #
    for scale in np.linspace(60, 140, 20)[::-1]:

        # Resize Image #
        image_resized = misc.imresize(image, int(scale))

        # Check if Resized Image is smaller than Template #
        if image_resized.shape[0] < template_hight or image_resized.shape[1] < template_width:
            break

        # Compute Template Matching #
        matched_image = cv2.matchTemplate(image_resized, template, method=cv2.TM_CCOEFF)
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(matched_image)

        # Store best match #
        if best_max_val is None or max_val > best_max_val:
            best_max_val = max_val
            best_template_match = matched_image

    # Resize best Match to Original Size #
    best_template_match = misc.imresize(best_template_match, image.shape)

    # Normalize array to Value 0-255 #
    cv2.normalize(best_template_match, best_template_match, 0, 255, cv2.NORM_MINMAX)

    # Return template matched picture #
    return best_template_match






