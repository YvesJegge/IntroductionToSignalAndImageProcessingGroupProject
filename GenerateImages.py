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



"""
-----------------------------------------------------------------------------------------------------
GenerateImages
-----------------------------------------------------------------------------------------------------
This File will generate images
-----------------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":

    # -- Set Parameter -- #
    startImage = 1                      # Start image1
    endImage = 23                       # End image (Do not exceed maximal number of images!)


    xMask = 200
    yMask = 300
    amountOfCutsPerImage = 150

    amountOfImages = endImage - startImage + 1

    # -- Print Out Message -- #
    print("Generate images \n==============================\n")

    # Generating Text File #
    f = open('data/templates/background/bg.txt', 'w')


    for ImageCount in range(startImage-1, endImage):

        # Generate image paths #
        image_path = "data/images_1/" + str(ImageCount + 1) + ".jpg"
        solution_path = "data/ground_truths/" + str(ImageCount + 1) + ".png"



        # Import image #
        img = plt.imread(image_path).astype(np.uint8)
        image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Get size of image
        xImage = image.shape[1]
        yImage = image.shape[0]

        # Import solution #
        solution = plt.imread(solution_path).astype(np.uint8)


        # Find contour of solution #
        ret,thresh = cv2.threshold(solution,0.5,1,cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bx,by,bw,bh = cv2.boundingRect(contours[0])


        for i in range(amountOfCutsPerImage):
            # Generate random x, y position of cut
            while True:
                xx = np.random.randint(0, xImage - xMask)
                yy = np.random.randint(0, yImage - yMask)

                # Make shure waldo is not in the cutted image
                if (solution[yy,xx] == 0) and (solution[yy + yMask,xx + xMask] == 0) and (image [yy,xx] > 0) and (image[yy + yMask,xx + xMask] > 0):
                    break


            # Cut waldo out and save this image #
            crop = image[yy:yy+yMask,xx:xx+xMask]
            plt.imsave("data/templates/background/" + str(ImageCount * amountOfCutsPerImage + i + 1) + ".png" , crop, cmap='gray', format='png')
            f.write("neg/"+ str(ImageCount * amountOfCutsPerImage + i + 1) + ".png"+"\n")


        # Status
        print('Image: ' + np.str(ImageCount + 1))

    # Close File
    f.close()
