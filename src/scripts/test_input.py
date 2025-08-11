import pyautogui
import time

print("The script will start in 3 seconds. Click on the RetroArch window!")
time.sleep(3)

# Simulates pressing the 'z' key (which we mapped to the A button)
# Mario should jump!
pyautogui.press('z')
print("Pressed 'z' (A Button). Did Mario jump?")

time.sleep(1)

# Simulates pressing the right arrow key
pyautogui.press('right')
print("Pressed 'right arrow'. Did Mario walk?")
