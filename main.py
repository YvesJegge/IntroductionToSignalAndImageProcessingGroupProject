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
This File wil controll/ test the group project Where is Waldo
-----------------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":

    # -- Print Out Message -- #
    print("Start where is Waldo-program ")

    fw.find_waldo(0)