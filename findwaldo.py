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
    image = color_matching(image)

    # Compute keypoint_detection #
    circle_matched_image = circle_matching(image)

    # Compute Template Matching
    template_matched_image_face = template_matching(image, "data/templates/WaldoFace.jpg")
    template_matched_image_glasses = template_matching(image, "data/templates/WaldoGlasses.jpg")

    # Put all results together #
    matched_image = np.uint16(template_matched_image_face) + np.uint16(template_matched_image_glasses)

    # Only for Testing Intensity Map #
    #display_denisty_map(image, matched_image)

    # Find Maximum Value of intensity Map #
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(matched_image)

    # Convert Coordinate Origin #
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
    plt.imshow(denisty_image, cmap='gray')

    plt.axis('on')
    plt.title('Intensity Map')
    plt.show
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
    black_filtered = np.uint8((image_hsv[:, :, 2] < 95))

    # Kernels #
    kernel_noise = np.ones((5,5))
    kernel_small = np.ones((4,3))
    kernel_big = np.ones((30,10))

    # Remove object with too small and too big size #
    red_filtered = remove_image_objects(red_filtered, 10, 250)
    white_filtered = remove_image_objects(white_filtered, 4, 60)
    pink_filtered = remove_image_objects(pink_filtered, 15, 3000)
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
    show_circle_in_image = True
    show_filtered_image = True
    window_high = 50
    window_width = 50

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
Method: line_matching()
------------------------------------------------------------------------------------------------------
This Method search for lines in the image (for searching the typically Wally Pattern).
This algorithm need first a preprocessingof the picture
------------------------------------------------------------------------------------------------------
Input  Parameter:       image as a input

Output Parameter:       New Image (Near Line there is the original picture, other black)
----------------------------------------------------------------------------------------------------*/
"""
def line_matching(image):
    # Settings for Line Matching #
    show_line = True

    # Convert to Gray Image #
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Compute Canny edges #
    edges = cv2.Canny(image_gray, 50, 150, apertureSize=3)

    # Finding Lines #
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=4, maxLineGap=2)

    # Draw Lines
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Show filtered image  #
    if show_line:
        plt.figure(400)
        plt.imshow(image)

    # Finding Lines #

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





