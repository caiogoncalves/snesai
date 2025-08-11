import cv2
import numpy as np
import pyautogui
from mss import mss
import time

# --- SETTINGS ---

# 1. Define the screen capture area (top, left, width, height)
#    Adjust these values for your RetroArch window
bounding_box = {'top': 100, 'left': 100, 'width': 800, 'height': 600}

# 2. Load the enemy template
#    Make sure the 'enemy_template.png' file is in the 'assets' folder
template = cv2.imread('assets/enemy_template.png', 0)
if template is None:
    raise FileNotFoundError("Could not find 'assets/enemy_template.png'. Make sure the file exists in the 'assets' folder.")

# Get the width and height of the template to draw the rectangle
w, h = template.shape[::-1]

# 3. Define the jump key
JUMP_KEY = 'z'  # The key you mapped to the A button in RetroArch

# --- BOT LOGIC ---

sct = mss()

print("The bot will start in 3 seconds...")
print("Click on the RetroArch window NOW!")
time.sleep(3)
print("Bot active! Press 'q' in the view window to stop.")

while True:
    # Capture the game screen
    sct_img = sct.grab(bounding_box)

    # Convert the image to a format that OpenCV understands (grayscale)
    frame = np.array(sct_img)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Search for the template (enemy) in the captured image
    # The value 0.8 is the "confidence threshold". You can adjust it (0.7-0.95).
    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where(res >= threshold)

    # Check if any match was found
    if np.any(loc[0]):
        print("Enemy detected! JUMPING!")

        # Press the jump key
        pyautogui.press(JUMP_KEY)

        # Draw a rectangle around the detected enemy in the view window
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            break # Draw only on the first enemy found to avoid clutter

        # A short pause to avoid pressing the key 60x per second
        time.sleep(0.1)

    # Show the bot's vision (with the rectangle if an enemy is found)
    cv2.imshow("Bot's Vision", frame)

    # Condition to stop the program
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
