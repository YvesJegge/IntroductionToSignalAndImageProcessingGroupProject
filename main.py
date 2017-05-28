"""
******************************************************************************************************
 Project           Introduction to Signal and Image Processing - Group Project: Where is Waldo?
 Filename          main.py

 Institution:      University of Bern

 Python            Python 3.6

 @author           Simon Scheurer, Yves Jegge
 @date             11.05.2016

 @status           Development

******************************************************************************************************
"""
# Import Package
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import signal
import matplotlib.patches as patches
import cv2

# Import Modul #
import findwaldo as fw
import ColorMatching as cm
import TemplateMatching as tm
import ShapeMatching as sm
import FaceMatching as fm

"""
-----------------------------------------------------------------------------------------------------
Main Programm
-----------------------------------------------------------------------------------------------------
This File wil control / test the group project Where is Waldo
-----------------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":

    # -- Set Parameter -- #
    startImage = 1                      # Start image1
    endImage = 23                           # End image (Do not exceed maximal number of images!)
    testOnlyMyFunction = False             # True: Test only my given function                        False: Test findwaldo()
    showImages = False                      # True: Show images                                        False: Only calculation
    showSubplot = False                    # True: Show images in subplot                             False: Show images separatly
    markTruePosition = True                # True: Mark true position of waldo                        False: Do not mark waldo
    showOnlyWrongPositionsImages = False   # True: Plot only images with wrong position calculation   False: Do not show true position
    cutWaldoOut = False                    # True: Cut waldo out and save image in \data\waldo        False: Do not store waldo

    xMask = 20
    yMask = 40

    amountOfImages = endImage - startImage + 1

    # -- Print Out Message -- #
    print("Start where is Waldo-program \n==============================\n")


    # -- Test function -- #

    amountCorrectPositions = 0
    for ImageCount in range(startImage-1, endImage):

        # Generate image paths #
        image_path = "data/images_1/" + str(ImageCount + 1) + ".jpg"
        solution_path = "data/ground_truths/" + str(ImageCount + 1) + ".png"

        # Import image #
        img = plt.imread(image_path).astype(np.uint8)

        # Import solution #
        solution = plt.imread(solution_path).astype(np.uint8)

        # Find contour of solution #
        ret,thresh = cv2.threshold(solution,0.5,1,cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bx,by,bw,bh = cv2.boundingRect(contours[0])

        if testOnlyMyFunction:
        ###########################################################################
        # Test only my function()
        ###########################################################################
            # Cut out waldo
            crop = img[by:by+bh,bx:bx+bw]

            # Test my function # ToDo: Insert your function here!
            #filtered_img = cm.color_matching(crop)
            filtered_img = tm.eye_matching(crop)
            #filtered_img = sm.shirt_cap_matching(img)
            #filtered_img = tm.template_matching(crop, "data/templates/WaldoFace.jpg")
            #filtered_img = tm.template_matching(crop, "data/templates/WaldoGlasses.jpg")
            #filtered_img =fm.FaceMatching(img)

            # Plot results
            plt.figure(400)
            plt.subplot(np.ceil(np.sqrt(amountOfImages)), np.ceil(np.sqrt(amountOfImages)), ImageCount + 1)
            plt.imshow(filtered_img, cmap='gray')
            plt.title('Image: ' + np.str(ImageCount + 1))

            # Status
            print('Image: ' + np.str(ImageCount + 1))


        else:
        ###########################################################################
        # Test findwaldo()
        ###########################################################################
            # Call function to test #
            x,y = fw.find_waldo(img)

            # Check position #
            if solution[img.shape[0]-y, x] > 0:
                positionCorrect = True
                amountCorrectPositions += 1

            else:
                positionCorrect = False


            # Cut waldo out and save this image #
            if cutWaldoOut == True:
                crop = img[by:by+bh,bx:bx+bw]
                plt.imsave("data/waldo/" + str(ImageCount + 1) + "_waldo.jpg",crop)


            # Show images #
            if (showImages == True) and ((showOnlyWrongPositionsImages == False) or (positionCorrect == False)):

                # Put images in subplot or separate figure #
                if showSubplot == True:
                    plt.subplot(np.ceil(np.sqrt(amountOfImages)), np.ceil(np.sqrt(amountOfImages)), ImageCount + 1)
                else:
                    plt.figure(ImageCount + 1)

                # Show image #
                mask = np.ones((img.shape[0], img.shape[1])).astype(np.uint8)
                mask[(img.shape[0]-y-yMask):(img.shape[0]-y+yMask), (x-xMask):(x+xMask)] = 0
                img -= (np.multiply(img, mask[:,:,None]) * 0.6).astype(np.uint8)


                # Mark wally
                if markTruePosition == True:
                    cv2.rectangle(img,(bx,by),(bx+bw,by+bh),(0,255,0),3)

                plt.imshow(img)
                plt.axis('off')

                # Set title of image #
                if positionCorrect == True:
                    plt.title('Image: ' + np.str(ImageCount + 1) + '  Correct')
                else:
                    plt.title('Image: ' + np.str(ImageCount + 1) + '  Wrong')

            # Status
            print('Image: ' + np.str(ImageCount + 1))

    # -- Show results -- #
    if testOnlyMyFunction == False:
        print("\n\nResults \n==============================\n")
        print(np.str(np.round(amountCorrectPositions / amountOfImages * 100)) + "% of positions are correct")
    plt.show()