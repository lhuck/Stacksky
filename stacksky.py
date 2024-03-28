
import numpy as np
import cv2

# Função para encontrar a homografia entre duas imagens usando correspondências de pontos
def findHomography(image_1_kp, image_2_kp, matches):
    # Inicializar arrays para pontos correspondentes
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    # Extrair pontos correspondentes das correspondências
    for i in range(0, len(matches)):
        image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
        image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

    # Verificar se há menos de 4 correspondências de pontos
    if len(matches) < 4:
        continue_process = input("Menos de 4 correspondências de pontos foram encontradas. Deseja continuar mesmo assim? (s/n): ")
        if continue_process.lower() != 's':
            exit()

    # Calcular a homografia usando RANSAC
    homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

    return homography 

# Função para alinhar imagens usando detector de características selecionado pelo usuário
def align_images(images):
    outimages = []

    # Perguntar ao usuário qual detector de características usar
    detector_type = input("Qual detector de características deseja usar? (SIFT: 1/ ORB: 2): ").upper()

    # Inicializar o detector de características com base na escolha do usuário
    if detector_type == "1":
        detector = cv2.SIFT_create()
    elif detector_type == "2":
        detector = cv2.ORB_create(nfeatures=1500, scaleFactor=1.01, nlevels=1, edgeThreshold=60)
    else:
        print("Opção inválida. Usando ORB como padrão.")
        detector = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31)

    # Detectar características da imagem base
    print("Detecting features of base image")
    outimages.append(images[0])
    image1gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    image_1_kp, image_1_desc = detector.detectAndCompute(image1gray, None)

    # Alinhar as imagens restantes em relação à imagem base
    for i in range(1, len(images)):
        print("alinhando imagem {}".format(i))
        image_i_kp, image_i_desc = detector.detectAndCompute(images[i], None)

        # Encontrar correspondências entre as características das imagens usando força bruta (BFMatcher)
        bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        image_1_desc = image_1_desc.astype(np.float32)
        image_i_desc = image_i_desc.astype(np.float32)
        rawMatches = bf.match(image_i_desc, image_1_desc)
        sortMatches = sorted(rawMatches, key=lambda x: x.distance)
        matches = sortMatches[0:128]

        # Calcular a homografia entre as imagens
        hom = findHomography(image_i_kp, image_1_kp, matches)
        newimage = cv2.warpPerspective(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_LINEAR)

        outimages.append(newimage)

    return outimages

# Função para computar o mapa de gradientes de uma imagem
def doLap(image):
    kernel_size = 5         # Tamanho da janela Laplaciana
    blur_size = 5           # Tamanho do kernel para o desfoque gaussiano

    # Aplicar desfoque gaussiano e calcular a Laplaciana
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)

# Função para empilhar imagens focalizadas
def focus_stack(unimages):
    # Alinhar as imagens
    images = align_images(unimages)

    # Calcular o mapa de gradientes das imagens
    print ("Calculando o laplaciano das imagens borradas")
    laps = []
    for i in range(len(images)):
        print ("Lap {}".format(i))
        laps.append(doLap(cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)))

    laps = np.asarray(laps)
    print ("Forma de conjunto de laplacianos = {}".format(laps.shape))

    # Encontrar os pontos de máxima nitidez em todas as imagens
    output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)
    abs_laps = np.absolute(laps)
    maxima = abs_laps.max(axis=0)
    bool_mask = abs_laps == maxima
    mask = bool_mask.astype(np.uint8)
    for i in range(0,len(images)):
        output = cv2.bitwise_not(images[i],output, mask=mask[i])

    return 255-output
