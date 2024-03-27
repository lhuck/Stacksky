import os
import cv2
import stacksky

def stackHDRs(image_files):
    focusimages = []
    for img in image_files:
        print("Lendo em arquivo {}".format(img))
        focusimages.append(cv2.imread("input/{}".format(img)))

    merged = stacksky.focus_stack(focusimages)

   
    
    #teste 1
    test = cv2.medianBlur(merged,3)
    #salva o arquivo do teste 1
    cv2.imwrite("test1.png", test)
    
     #test 2
    imagem_dst = cv2.fastNlMeansDenoisingColored(merged,None,7,21,7,21)
    #salva o arquivo do teste 2 
    cv2.imwrite("test2.png", imagem_dst)
    
    #salva o arquivo original 
    cv2.imwrite("merged.png", merged)


if __name__ == "__main__":
    image_files = sorted(os.listdir("input"))
    for img in image_files:
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_files.remove(img)

    continue_process = input("Menos de 4 correspondências de pontos foram encontradas.\npode ser que a imagem nao saia com uma qualidade bom boa \n Deseja continuar mesmo assim? (y/n): ")
    if continue_process.lower() != 'y':
        exit()

    stackHDRs(image_files)
    
    print("pronto!!!")
    