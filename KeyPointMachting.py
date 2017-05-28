"""
******************************************************************************************************
 Project           Introduction to Signal and Image Processing - Group Project: Where is Waldo?
 Filename          KeyPointMatching.py

 Institution:      University of Bern

 Python            Python 3.6

 @author           Simon Scheurer, Yves Jegge
 @date             28.05.2016

 @status           Development

******************************************************************************************************
"""
# Import Package #
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import Modul #

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
def keypoint_detection():

    # -- Set Settings for template Matching-- #
    canny_detection = False
    blur_filter = False

    # Load Files #
    img = cv2.imread('data/images_1/4.jpg')
    img2 = cv2.imread('data/waldo/2_waldo.jpg')

    # Convert to Gray-Image #
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Filtering the Image #
    if blur_filter:
        img1 = cv2.medianBlur(img1, 5)
        img2 =cv2.medianBlur(img2, 5)

    # Edge detection for Template #
    if canny_detection:
        # Compute for Canny edge dedection for image #
        max_magnitude_image = np.median(img1)
        thr_high_image = 0.3 * max_magnitude_image
        thr_low_image = thr_high_image / 2
        image = cv2.Canny(img1, threshold1=thr_low_image, threshold2=thr_high_image)
        # Compute for Canny edge dedection for template #
        max_magnitude_template = np.median(img2)
        thr_high_template = 0.2 * max_magnitude_template
        thr_low_template = thr_high_template / 2
        template = cv2.Canny(img2, threshold1=thr_low_template, threshold2=thr_high_template)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=10)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()





"""
-----------------------------------------------------------------------------------------------------
Main Programm
-----------------------------------------------------------------------------------------------------
This File wil control / test the group project Where is Waldo
-----------------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":
    keypoint_detection()

