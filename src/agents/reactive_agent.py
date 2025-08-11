"""
A reactive agent that plays a SNES game by detecting enemies and jumping.

This agent uses template matching to find enemies on the screen and then
simulates a key press to make the character jump. The agent's configuration
is loaded from the `src/config.py` file.
"""

import cv2
import numpy as np
import pyautogui
from mss import mss
import time
import os
import sys

<<<<<<< HEAD
# Add the 'src' directory to the path to import the config module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import BOUNDING_BOX, JUMP_KEY, ENEMY_TEMPLATE_PATH, CONFIDENCE_THRESHOLD


def load_template(path):
    """
    Loads the enemy template image from the given path.

    Args:
        path (str): The path to the template image.

    Returns:
        numpy.ndarray: The template image in grayscale.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find template image at: {path}")
    template = cv2.imread(path, 0)
    if template is None:
        raise IOError(f"Could not read template image at: {path}")
    return template

=======
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
>>>>>>> a54129cf58c1ab5fe6d5ef1f3c03ba1870c41dda

def capture_screen(bounding_box):
    """
    Captures the game screen within the given bounding box.

<<<<<<< HEAD
    Args:
        bounding_box (dict): A dictionary with 'top', 'left', 'width', and 'height' of the screen area to capture.

    Returns:
        numpy.ndarray: The captured screen as a NumPy array.
    """
    with mss() as sct:
        sct_img = sct.grab(bounding_box)
        return np.array(sct_img)


def detect_enemy(frame, template, threshold):
    """
    Detects enemies in the frame using template matching.

    Args:
        frame (numpy.ndarray): The current game screen.
        template (numpy.ndarray): The enemy template image.
        threshold (float): The confidence threshold for template matching.

    Returns:
        tuple: A tuple containing the locations of the detected enemies and the width and height of the template.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
=======
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
>>>>>>> a54129cf58c1ab5fe6d5ef1f3c03ba1870c41dda
    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
<<<<<<< HEAD
    w, h = template.shape[::-1]
    return loc, w, h


def jump(key):
    """
    Simulates a key press to make the character jump.

    Args:
        key (str): The key to press for jumping.
    """
    pyautogui.press(key)


def main():
    """
    The main function that runs the reactive agent.
    """
    try:
        template = load_template(ENEMY_TEMPLATE_PATH)
        print("The robot will start in 3 seconds...")
        print("Click on the RetroArch window NOW!")
        time.sleep(3)
        print("Robot active! Press 'q' in the preview window to stop.")

        while True:
            frame = capture_screen(BOUNDING_BOX)
            loc, w, h = detect_enemy(frame, template, CONFIDENCE_THRESHOLD)

            if np.any(loc[0]):
                print("Enemy detected! JUMPING!")
                jump(JUMP_KEY)

                # Draw a rectangle around the detected enemy in the preview window
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                    break  # Draw only on the first enemy found to avoid clutter

                # A short pause to avoid pressing the key 60x per second
                time.sleep(0.1)

            # Show the robot's vision (with the rectangle if an enemy is found)
            cv2.imshow('Robot Vision', frame)

            # Condition to stop the program
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    except (FileNotFoundError, IOError) as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
=======

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
>>>>>>> a54129cf58c1ab5fe6d5ef1f3c03ba1870c41dda
