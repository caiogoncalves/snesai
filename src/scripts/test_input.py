import pyautogui
import time

print("O script começará em 3 segundos. Clique na janela do RetroArch!")
time.sleep(3)

# Simula o pressionamento da tecla 'z' (que mapeamos para o botão A)
# Mario deve pular!
pyautogui.press('z') 
print("Pressionou 'z' (Botão A). Mario pulou?")

time.sleep(1)

# Simula o pressionamento da seta para a direita
pyautogui.press('right')
print("Pressionou 'seta para a direita'. Mario andou?")