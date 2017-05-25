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
from scipy import signal, misc
import cv2

# Import Modul #

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
