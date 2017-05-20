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

    # Compute Template Matching
    template_matched_image = template_matching(image, "data/templates/WaldoSmall.jpeg")

    # Compute keypoint_detection #
    # (Maybe better than Template Matching, however not yet implemented) #
    #template_matched_image = keypoint_detection(image, "data/templates/WaldoSmall.jpeg")

    # Only for Testing Intensity Map #
    #display_denisty_map(image, template_matched_image)

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
Method: color_matching()
------------------------------------------------------------------------------------------------------
This Method takes the original image and find via color matching waldo
------------------------------------------------------------------------------------------------------
Input  Parameter:       original_image

Output Parameter:       Give probability map back with the same size of original image (0....1)
----------------------------------------------------------------------------------------------------*/
"""
def color_matching(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_filter = np.bitwise_and(np.bitwise_and(np.bitwise_or((image_hsv[:, :, 0] < 5), (image_hsv[:, :, 0] > 175)),(image_hsv[:, :, 1] > 128)),(image_hsv[:, :, 2] > 128))

    print(np.min(image_hsv[:, :, 0]))
    print(np.max(image_hsv[:, :, 0]))

    plt.imshow(image_filter, cmap = "gray")
    plt.axis('off')
    plt.show()




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
    # Set Settings for template Matching #
    gray_picture = True
    canny_detection = False

    # Read in Template Picture #
    template = cv2.imread(template_path)

    # Convert Image and template to Gray-Scale #
    if gray_picture:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

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

"""
/*----------------------------------------------------------------------------------------------------
Method: keypoint_detection()
------------------------------------------------------------------------------------------------------
This Method runs a keypoint detection algorithm to match a template to a picture
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input, template path

Output Parameter:       Density image that is generated from the template matching
----------------------------------------------------------------------------------------------------*/
"""
def keypoint_detection(image, template_path):

    # Set Settings for template Matching #
    canny_detection = False
    gray_picture = False

    # Read in Template Picture #
    template = cv2.imread(template_path)

    # Convert Image and template to Gray-Scale #
    if gray_picture:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Edge detection for #
    if canny_detection:
        template = cv2.Canny(template, threshold1=50, threshold2=200)
        image = cv2.Canny(image, threshold1=50, threshold2=200)

    # Keypoin dedection Algorithm #
    # To be implemented ... #
    best_template_match = image

    # Normalize array to Value 0-255 #
    cv2.normalize(best_template_match, best_template_match, 0, 255, cv2.NORM_MINMAX)

    # Return template matched picture #
    return best_template_match





