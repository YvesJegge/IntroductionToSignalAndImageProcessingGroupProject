"""
******************************************************************************************************
 Project           Introduction to Signal and Image Processing - Group Project: Where is Waldo?
 Filename          TemplateMatching.py

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
from scipy import signal, misc, ndimage
import cv2

# Import Modul #

"""
/*----------------------------------------------------------------------------------------------------
Method: template_matching()
------------------------------------------------------------------------------------------------------
This Method runs a template matching algorithm throws the image
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input, template path, scale parameters

Output Parameter:       Density image that is generated from the template matching
----------------------------------------------------------------------------------------------------*/
"""
def template_matching(image, template_path, scale_min = 50, scale_max = 100, amount_of_scale = 3, gray_picture = True):

    #-- Set Settings for template Matching-- #
    canny_detection = False

    # Read in Template Picture #
    template = plt.imread(template_path)

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
        thr_high_template = 0.3 * max_magnitude_template
        thr_low_template = thr_high_template / 2
        template = cv2.Canny(template, threshold1=thr_low_template, threshold2=thr_high_template)

    # Initialize used variable #
    dentency_map = np.zeros((image.shape[0],image.shape[1]))
    (template_hight, template_width) = template.shape[:2]

    # -- Loop over the scales of the image -- #
    for scale in np.linspace(scale_min, scale_max, amount_of_scale)[::-1]:

        # Resize Image #
        image_resized = misc.imresize(image, int(scale))

        # Check if Resized Image is smaller than Template #
        if image_resized.shape[0] < template_hight or image_resized.shape[1] < template_width:
            break

        # Compute Template Matching #
        matched_image = cv2.matchTemplate(image_resized, template, method=cv2.TM_CCOEFF)
        matched_image = cv2.normalize(matched_image, matched_image, 0, 255, cv2.NORM_MINMAX)

        # Resize image to original size
        resized_image = misc.imresize(matched_image, image.shape)

        # sum up max-values #
        d = (resized_image > dentency_map)
        dentency_map[d] = resized_image[d]

    # Normalize array to Value 0-255 #
    dentency_map = ndimage.maximum_filter(dentency_map, (30,20))
    filtered_img = dentency_map# = cv2.normalize(dentency_map, dentency_map, 0, 255, cv2.NORM_MINMAX)
    
    # Return template matched picture #
    return filtered_img

"""
/*----------------------------------------------------------------------------------------------------
Method: eye_matching()
------------------------------------------------------------------------------------------------------
This Method search the eye of waldo
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input

Output Parameter:       Density image that is generated from the eye matching
----------------------------------------------------------------------------------------------------*/
"""
def eye_matching(image):

    amount_of_templates = 3
    dentency_map = np.zeros((image.shape[0], image.shape[1]))

    # Use different eye templates
    for ii in range(0, amount_of_templates):

        dentency = template_matching(image, ("data/templates/Glasses/" + str(ii + 1) + "_Glasses.jpg"))

        # sum up max-values #
        d = (dentency > dentency_map)
        dentency_map[d] = dentency[d]

    dentency_map = cv2.normalize(dentency_map, dentency_map, 0, 255, cv2.NORM_MINMAX)

    return dentency_map

"""
/*----------------------------------------------------------------------------------------------------
Method: hair_matching()
------------------------------------------------------------------------------------------------------
This Method search the hair close to the cap of waldo
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input

Output Parameter:       Density image that is generated from the eye matching
----------------------------------------------------------------------------------------------------*/
"""
def hair_matching(image):

    amount_of_templates = 3
    dentency_map = np.zeros((image.shape[0], image.shape[1]))

    # Use different eye templates
    for ii in range(0, amount_of_templates):

        dentency = template_matching(image, ("data/templates/Hair/" + str(ii + 1) + "_Hair.jpg"), 75, 100, 2)

        # sum up max-values #
        d = (dentency > dentency_map)
        dentency_map[d] = dentency[d]

    dentency_map = cv2.normalize(dentency_map, dentency_map, 0, 255, cv2.NORM_MINMAX)

    return dentency_map

"""
/*----------------------------------------------------------------------------------------------------
Method: hairfront_matching()
------------------------------------------------------------------------------------------------------
This Method search the top of the hair of waldo
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input

Output Parameter:       Density image that is generated from the eye matching
----------------------------------------------------------------------------------------------------*/
"""
def hairfront_matching(image):

    amount_of_templates = 3
    dentency_map = np.zeros((image.shape[0], image.shape[1]))

    # Use different eye templates
    for ii in range(0, amount_of_templates):

        dentency = template_matching(image, ("data/templates/HairFront/" + str(ii + 1) + "_HairFront.jpg"), 75, 100, 2)

        # sum up max-values #
        d = (dentency > dentency_map)
        dentency_map[d] = dentency[d]

    dentency_map = cv2.normalize(dentency_map, dentency_map, 0, 255, cv2.NORM_MINMAX)

    return dentency_map