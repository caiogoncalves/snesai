import cv2
import numpy as np
import pyautogui
from mss import mss
import time

# --- CONFIGURAÇÕES ---

# 1. Defina a área de captura da tela do jogo (top, left, width, height)
#    Ajuste estes valores para a janela do seu RetroArch
bounding_box = {'top': 100, 'left': 100, 'width': 800, 'height': 600}

# 2. Carregue o template do inimigo
#    Certifique-se que o ficheiro 'inimigo_template.png' está na mesma pasta
template = cv2.imread('inimigo_template.png', 0)
if template is None:
    raise FileNotFoundError("Não foi possível encontrar 'inimigo_template.png'. Certifique-se que o ficheiro existe.")

# Obtém a altura e largura do template para desenhar o retângulo
w, h = template.shape[::-1]

# 3. Defina a tecla de pulo
PULO_TECLA = 'z'  # A tecla que você mapeou para o botão A no RetroArch

# --- LÓGICA DO ROBÔ ---

sct = mss()

print("O robô vai começar em 3 segundos...")
print("Clique na janela do RetroArch AGORA!")
time.sleep(3)
print("Robô ativo! Pressione 'q' na janela de visualização para parar.")

while True:
    # Captura a tela do jogo
    sct_img = sct.grab(bounding_box)
    
    # Converte a imagem para um formato que o OpenCV entende (escala de cinza)
    frame = np.array(sct_img)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Procura pelo template (inimigo) na imagem capturada
    # O valor 0.8 é o "threshold de confiança". Pode ajustá-lo (0.7-0.95).
    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where(res >= threshold)
    
    # Verifica se encontrou alguma correspondência
    if np.any(loc[0]):
        print("Inimigo detetado! A PULAR!")
        
        # Pressiona a tecla de pulo
        pyautogui.press(PULO_TECLA)
        
        # Desenha um retângulo à volta do inimigo detetado na janela de visualização
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            break # Desenha apenas no primeiro inimigo encontrado para não poluir
        
        # Uma pequena pausa para não pressionar a tecla 60x por segundo
        time.sleep(0.1)

    # Mostra a visão do robô (com o retângulo se um inimigo for encontrado)
    cv2.imshow('Visão do Robô', frame)

    # Condição para parar o programa
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break