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

# Import Modul #
import findwaldo as fw


"""
-----------------------------------------------------------------------------------------------------
Main Programm
-----------------------------------------------------------------------------------------------------
This File wil control / test the group project Where is Waldo
-----------------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":

    # -- Set Parameter -- #
    amountOfImages = 2
    showImages = True
    showSubplot = False
    showOnlyWrongPositions = False
    xMask = 20
    yMask = 40

    # -- Print Out Message -- #
    print("Start where is Waldo-program \n==============================\n")


    # -- Test function -- #

    amountCorrectPositions = 0
    for ImageCount in range(amountOfImages):

        # Generate image paths #
        image_path = "data/images/" + str(ImageCount + 1) + ".jpg"
        solution_path = "data/ground_truths/" + str(ImageCount + 1) + ".png"

        # Import image #
        img = plt.imread(image_path).astype(np.uint8)

        # Import solution #
        solution = plt.imread(solution_path).astype(np.uint8)

        # Call function to test #
        x,y = fw.find_waldo(img)

        # Check position #
        if solution[img.shape[0]-y, x] > 0:
            positionCorrect = True
            amountCorrectPositions += 1

        else:
            positionCorrect = False


        # Show images #
        if (showImages == True) and ((showOnlyWrongPositions == False) or (positionCorrect == False)):

            # Put images in subplot or separate figure #
            if showSubplot == True:
                plt.subplot(np.ceil(np.sqrt(amountOfImages)), np.ceil(np.sqrt(amountOfImages)), ImageCount + 1)
            else:
                plt.figure(ImageCount + 1)

            # Show image #
            mask = np.ones((img.shape[0], img.shape[1])).astype(np.uint8)
            mask[(img.shape[0]-y-yMask):(img.shape[0]-y+yMask), (x-xMask):(x+xMask)] = 0
            img -= (np.multiply(img, mask[:,:,None]) * 0.6).astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

            # Set title of image #
            if positionCorrect == True:
                plt.title('Image: ' + np.str(ImageCount + 1) + '  Correct')
            else:
                plt.title('Image: ' + np.str(ImageCount + 1) + '  Wrong')



    # -- Show results -- #
    print("\n\nResults \n==============================\n")
    print(np.str(np.round(amountCorrectPositions / amountOfImages * 100)) + "% of positions are correct")
    plt.show()