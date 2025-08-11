"""
A simple script to display the game screen captured from the emulator.

This script continuously captures the screen area defined in `src/config.py`
and displays it in a window using OpenCV.
"""

import numpy as np
import cv2
from mss import mss
import os
import sys

# Add the 'src' directory to the path to import the config module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import BOUNDING_BOX


def view_game_screen():
    """
    Captures and displays the game screen in real-time.
    """
    sct = mss()

    print("Press 'q' in the image window to exit.")

    try:
        while True:
            # Capture the screen in the defined area
            sct_img = sct.grab(BOUNDING_BOX)
            
            # Convert the image to a format that OpenCV understands
            img = np.array(sct_img)
            
            # Display the image in a window
            cv2.imshow('Robot Vision', img)
            
            # Wait for 1ms. If 'q' is pressed, the loop breaks.
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    view_game_screen()