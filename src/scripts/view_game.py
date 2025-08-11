import numpy as np
import cv2
from mss import mss

# Substitua com as coordenadas que você encontrou
# Exemplo: {'top': 100, 'left': 100, 'width': 800, 'height': 600}
bounding_box = {'top': 100, 'left': 100, 'width': 800, 'height': 600}

sct = mss()

print("Pressione 'q' na janela da imagem para sair.")

while True:
    # Captura a tela na área definida
    sct_img = sct.grab(bounding_box)
    
    # Converte a imagem para um formato que o OpenCV entende
    img = np.array(sct_img)
    
    # Exibe a imagem em uma janela
    cv2.imshow('Visão do Robô', img)
    
    # Espera por 1ms. Se 'q' for pressionado, o loop quebra.
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break