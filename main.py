import os
import cv2
import numpy as np
import stacksky

def stackHDRs(image_files):
    focusimages = []
    for img in image_files:
        print("Lendo em arquivo {}".format(img))
        focusimages.append(cv2.imread("input/{}".format(img)))

    merged = stacksky.focus_stack(focusimages)
    
    #salva o arquivo do metodo 
    cv2.imwrite("imagen_do_metodo.png", merged)
    print(" arquivo do metodo criada")

    #teste 1
    test = cv2.medianBlur(merged,3)
    #salva o arquivo do teste 1
    cv2.imwrite("test1.png", test)
    print(" arquivo do teste 1 criada")
    
    #test 2
    test2 = cv2.fastNlMeansDenoisingColored(merged,None,7,21,7,21)
    #salva o arquivo do teste 2 
    cv2.imwrite("test2.png", test2)
    print(" arquivo do teste 2 criada")
    
  
    #retorna a imagem blur
    blur = cv2.fastNlMeansDenoisingColored(merged,None,7,21,7,21)

    image1 = cv2.imread('imagen_do_metodo.png')
    # Criar uma nova imagem com a média dos canais de cor
    average_color = np.mean(image1, axis=(0, 1)).astype(int)
    layer_media = np.full_like(image1, average_color, dtype=np.uint8)

    cv2.imwrite("layer_media.png", layer_media)
    print(" arquivo do layer media criada")


    resultado = cv2.subtract(merged, layer_media)
    #salvar o resultado em um arquivo de imagem
    cv2.imwrite('resultado_blur.png', blur)
    print(" arquivo do blur criada")

    cv2.imwrite('resultado_subtracao.png', resultado)
    print(" arquivo da imagem final subitraida criada ")

if __name__ == "__main__":
    image_files = sorted(os.listdir("input"))
    for img in image_files:
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_files.remove(img)

 #   continue_process = input("Menos de 4 correspondências de pontos foram encontradas.\npode ser que a imagem nao saia com uma qualidade bom boa \n Deseja continuar mesmo assim? (y/n): ")
 #   if continue_process.lower() != 'y':
 #       exit()

    stackHDRs(image_files)
    
    print("pronto!!!")
   
