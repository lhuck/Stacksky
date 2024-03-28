import cv2
import numpy as np
import stacksky


image1 = cv2.imread('imagen_do_metodo.png')

# Calcular a média dos canais de cor da imagem
average_color = np.mean(image1, axis=(0, 1)).astype(int)

# Criar uma nova imagem com a média dos canais de cor
layer_media = np.full_like(image1, average_color, dtype=np.uint8)

# Exibir a imagem com a camada da média
cv2.imshow('Camada com Media da Imagem', layer_media)
cv2.imwrite("layer_media.png", layer_media)
cv2.waitKey(0)
cv2.destroyAllWindows()