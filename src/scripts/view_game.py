import numpy as np
import cv2
from mss import mss

# Replace with the coordinates you found
# Example: {'top': 100, 'left': 100, 'width': 800, 'height': 600}
bounding_box = {'top': 100, 'left': 100, 'width': 800, 'height': 600}

sct = mss()

print("Press 'q' in the image window to exit.")

while True:
    # Capture the screen in the defined area
    sct_img = sct.grab(bounding_box)

    # Convert the image to a format that OpenCV understands
    img = np.array(sct_img)

    # Display the image in a window
    cv2.imshow("Bot's Vision", img)

    # Wait for 1ms. If 'q' is pressed, the loop breaks.
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
