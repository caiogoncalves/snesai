"""
A simple script to test input simulation to the RetroArch emulator.

This script simulates pressing the 'z' key (mapped to button A) and the 'right' arrow key
to check if the emulator is receiving inputs correctly.
"""

import pyautogui
import time

# Define the keys to be pressed
JUMP_KEY = 'z'
RIGHT_KEY = 'right'

print("The script will start in 3 seconds. Click on the RetroArch window!")
time.sleep(3)

# Simulate pressing the 'z' key (mapped to button A)
# Mario should jump!
pyautogui.press(JUMP_KEY)
print(f"Pressed '{JUMP_KEY}' (Button A). Did Mario jump?")

time.sleep(1)

# Simulate pressing the right arrow key
pyautogui.press(RIGHT_KEY)
print(f"Pressed '{RIGHT_KEY}' (Right Arrow). Did Mario walk?")