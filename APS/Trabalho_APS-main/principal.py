# pyuic5 designe.ui -o principal.py

from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from matplotlib import pyplot as plt
# import mahotas
import time
from tqdm import tqdm
import cv2
import os
import sys
import numpy as np



def carregaImage():
    arquivo = QtWidgets.QFileDialog.getOpenFileName()[0]
    # INICIO MODULO 1

    def loading():
        for i in tqdm(range(100)):
            principal.progressBar.setHidden(False)
            time.sleep(0.01)
            principal.progressBar.setValue(i+1)
            if i == 99:
                principal.progressBar.setHidden(True)

        principal.progressBar.setValue(0)

    def largura_altura():
        imagem = cv2.imread(arquivo)
        loading()

        principal.largura.setText('Largura em pixels: '+str(imagem.shape[1]))
        principal.altura.setText('Altura em pixels: '+str(imagem.shape[0]))
        principal.canais.setText('Qtde de canais: '+str(imagem.shape[2])) 
        cv2.imshow("Imagem Seleção Brasileira", imagem) 
        cv2.waitKey(0)
    
    def valor_cores():
        imagem = cv2.imread(arquivo)
        loading()

        (b, g, r) = imagem[0, 0] 
        principal.pixel.setText('O pixel (0, 0) tem as seguintes cores:')
        principal.valorCores.setText('Vermelho: '+ str(r) + ' | ' + 'Verde: '+ str(g) + ' | ' + 'Azul: ' + str(b))
        
    
    def azul():
        imagem = cv2.imread(arquivo)
        loading()

        for y in range(0, imagem.shape[0]):
            for x in range(0, imagem.shape[1]):
                imagem[y, x] = (255,0,0)
        cv2.imshow("Imagem modificada", imagem)
        cv2.waitKey(0)
    
    def gradiente():
        imagem = cv2.imread(arquivo)
        loading()

        for y in range(0, imagem.shape[0]):
            for x in range (0, imagem.shape[1]):
                imagem[y, x] = (x%256, y%256, x%256)
        cv2.imshow("Imagem modificada", imagem)
        cv2.waitKey(0)
    
    def verde():
        loading()
        imagem = cv2.imread(arquivo)

        for y in range(0, imagem.shape[0], 1): 
            for x in range(0, imagem.shape[1], 1): 
                imagem[y, x] = (0,(x*y)%256,0)
        cv2.imshow("Imagem modificada", imagem)
        cv2.waitKey(0)

    def amarelo():
        imagem = cv2.imread(arquivo)
        loading()

        for y in range(0, imagem.shape[0], 10): 
            for x in range(0, imagem.shape[1], 10): 
                imagem[y:y+5, x: x+5] = (0,255,255)

        cv2.imshow("Imagem modificada", imagem)
        cv2.waitKey(0)

    principal.largura_altura.clicked.connect(largura_altura)
    principal.cores.clicked.connect(valor_cores)
    principal.azul.clicked.connect(azul)
    principal.gradiente.clicked.connect(gradiente)
    principal.verde.clicked.connect(verde)
    principal.verde_2.clicked.connect(amarelo)
    # FIM MODULO 1

    # INICIO MODULO 2

    def slicing():
        imagem = cv2.imread(arquivo)
        loading()

        imagem[30:50, :] = (255, 0, 0)
        imagem[100:150, 50:100] = (0, 0, 255)
        imagem[:, 200:220] = (0, 255, 255)
        imagem[150:300, 250:350] = (0, 255, 0)
        imagem[300:400, 50:150] = (255, 255, 0)
        imagem[250:350, 300:400] = (255, 255, 255)
        imagem[70:100, 300: 450] = (0, 0, 0)

        cv2.imshow("Imagem alterada", imagem)
        cv2.imwrite("Imagem alterada.jpg", imagem)
        cv2.waitKey(0)

    def desenho():
        imagem = cv2.imread(arquivo)
        loading()

        vermelho = (0, 0, 255)
        verde = (0, 255, 0)
        azul = (255, 0, 0)

        cv2.line(imagem, (0, 0), (100, 200), verde)
        cv2.line(imagem, (300, 200), (150, 150), vermelho, 5)
        cv2.rectangle(imagem, (20, 20), (120, 120), azul, 10)
        cv2.rectangle(imagem, (200, 50), (225, 125), verde, -1)
        (X, Y) = (imagem.shape[1] // 2, imagem.shape[0] // 2)

        for raio in range(0, 175, 15):
            cv2.circle(imagem, (X, Y), raio, vermelho)

        cv2.imshow("Desenhando sobre a imagem", imagem)
        cv2.waitKey(0)
    
    def texto():
        imagem = cv2.imread(arquivo)
        loading()

        fonte = cv2.FONT_HERSHEY_SIMPLEX
        texto = principal.textImg.text()
        cv2.putText(imagem,texto,(15,65), fonte, 
        2,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow("Texto sobre a Imagem", imagem)
        cv2.waitKey(0)

    def crop():
        imagem = cv2.imread(arquivo)
        loading()

        recorte = imagem[100:200, 100:200]
        cv2.imshow("Recorte da imagem", recorte)
        cv2.imwrite("recorte.jpg", recorte) 
        cv2.waitKey(0)

    def resize():
        imagem = cv2.imread(arquivo)
        loading()

        cv2.imshow("Original", imagem)
        proporcao = 100.0 / imagem.shape[1] 
        tamanho_novo = (100, int(imagem.shape[0] * proporcao))
        img_redimensionada = cv2.resize(imagem, tamanho_novo, 
        interpolation = cv2.INTER_AREA)
        16
        cv2.imshow("Imagem redimensionada", img_redimensionada)
        cv2.waitKey(0)

    def flip():
        imagem = cv2.imread(arquivo)
        loading()

        cv2.imshow("Original", imagem)
        img_redimensionada = imagem[::2,::2]
        cv2.imshow("Imagem redimensionada", img_redimensionada)
        cv2.waitKey(0)

    def rotate():
        imagem = cv2.imread(arquivo)
        loading()

        cv2.imshow("Original", imagem)
        flip_horizontal = imagem[::-1,:] 
        cv2.imshow("Flip Horizontal", flip_horizontal)
        flip_vertical = imagem[:,::-1] 
        cv2.imshow("Flip Vertical", flip_vertical)
        flip_hv = imagem[::-1,::-1]
        cv2.imshow("Flip Horizontal e Vertical", flip_hv)
        cv2.waitKey(0)

    def mascara1():
        imagem = cv2.imread(arquivo)
        loading()

        (alt, lar) = imagem.shape[:2] #captura altura e largura
        centro = (lar // 2, alt // 2) #acha o centro
        M = cv2.getRotationMatrix2D(centro, 30, 1.0) #30 graus
        img_rotacionada = cv2.warpAffine(imagem, M, (lar, alt))
        cv2.imshow("Imagem rotacionada em 45 graus", img_rotacionada)
        cv2.waitKey(0)

    def mascara2():
        imagem = cv2.imread(arquivo)
        loading()

        cv2.imshow("Original", imagem)
        mascara = np.zeros(imagem.shape[:2], dtype = "uint8")
        (cX, cY) = (imagem.shape[1] // 2, imagem.shape[0] // 2)
        cv2.circle(mascara, (cX, cY), 100, 255, -1)
        img_com_mascara = cv2.bitwise_and(imagem, imagem, mask = mascara)
        cv2.imshow("Máscara aplicada à imagem", img_com_mascara)
        cv2.waitKey(0)

    def mascara3():
        imagem = cv2.imread(arquivo)
        loading()
        
        cv2.imshow("Original", imagem)
        mascara = np.zeros(imagem.shape[:2], dtype = "uint8")
        (cX, cY) = (imagem.shape[1] // 2, imagem.shape[0] // 2)
        cv2.circle(mascara, (cX, cY), 180, 255, 70)
        cv2.circle(mascara, (cX, cY), 70, 255, -1)
        img_com_mascara = cv2.bitwise_and(imagem, imagem, mask = mascara)
        cv2.imshow("Máscara aplicada à imagem", img_com_mascara)
        cv2.waitKey(0)

    principal.slicing.clicked.connect(slicing)
    principal.imgDesenho.clicked.connect(desenho)
    principal.textoImage.clicked.connect(texto)
    principal.crop.clicked.connect(crop)
    principal.resize.clicked.connect(resize)
    principal.flip.clicked.connect(flip)
    principal.rotate.clicked.connect(rotate)
    principal.oneMascara.clicked.connect(mascara1)
    principal.twoMascara.clicked.connect(mascara2)
    principal.tresMascara.clicked.connect(mascara3)

    # FIM MODULO 3

    # INICIO MODULO 4
    def trocaCor():
        imagem = cv2.imread(arquivo)
        loading()
        
        cv2.imshow("Original", imagem)
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray", gray)
        hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV", hsv)
        lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
        cv2.imshow("L*a*b*", lab)
        cv2.waitKey(0)

    def canaisImage():
        imagem = cv2.imread(arquivo)
        loading()

        (canalAzul, canalVerde, canalVermelho) = cv2.split(imagem)
        cv2.imshow("Vermelho", canalVermelho)
        cv2.imshow("Verde", canalVerde)
        cv2.imshow("Azul", canalAzul)
        cv2.waitKey(0)

    def rgb():
        imagem = cv2.imread(arquivo)
        loading()

        (canalAzul, canalVerde, canalVermelho) = cv2.split(imagem)
        zeros = np.zeros(imagem.shape[:2], dtype = "uint8")
        cv2.imshow("Vermelho", cv2.merge([zeros, zeros, 
        canalVermelho]))
        cv2.imshow("Verde", cv2.merge([zeros, canalVerde, zeros]))
        cv2.imshow("Azul", cv2.merge([canalAzul, zeros, zeros]))
        cv2.imshow("Original", imagem)
        cv2.waitKey(0)

    def bgr(): #NÃO FUNCIONA
        img_bgr = cv2.imread(arquivo, 1) 
        cv2.imshow("Imagem Colorida", img_bgr)
        loading()
   
        color = ('b', 'g', 'r') 
        
        for i, col in enumerate(color): 
            histr = cv2.calcHist([img_bgr], [i], None, [256], [0, 256]) 
            plt.plot(histr, color = col) 
            plt.xlim([0, 256]) 
            
        plt.show() 
    
    def grafico1():
        imagem = cv2.imread(arquivo)
        loading()

        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) 
        cv2.imshow("Imagem P&B", imagem)

        h = cv2.calcHist([imagem], [0], None, [256], [0, 256])
        plt.figure()
        plt.title("Histograma P&B")
        plt.xlabel("Intensidade")
        plt.ylabel("Qtde de Pixels")
        plt.plot(h)
        plt.xlim([0, 256])
        plt.show()
        cv2.waitKey(0)
    
    def grafico2():
        imagem = cv2.imread(arquivo)
        loading()

        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        h_eq = cv2.equalizeHist(imagem)
        plt.figure()
        plt.title("Histograma Equalizado")
        plt.xlabel("Intensidade")
        plt.ylabel("Qtde de Pixels")
        plt.hist(h_eq.ravel(), 256, [0,256])
        plt.xlim([0, 256])
        plt.show()
        plt.figure()
        plt.title("Histograma Original")
        plt.xlabel("Intensidade")
        plt.ylabel("Qtde de Pixels")
        plt.hist(imagem.ravel(), 256, [0,256])
        plt.xlim([0, 256])
        plt.show()
        cv2.waitKey(0)

    def suavidade1():
        imagem = cv2.imread(arquivo)
        loading()

        imagem = imagem[::2,::2] # Diminui a imagem
        suave = np.vstack([
        np.hstack([imagem, cv2.blur(imagem, ( 3, 3))]), 
        np.hstack([cv2.blur(imagem, (5,5)), cv2.blur(imagem, ( 7, 7))]), 
        np.hstack([cv2.blur(imagem, (9,9)), cv2.blur(imagem, (11, 11))]), 
        ])
        cv2.imshow("Imagens suavisadas (Blur)", suave)
        cv2.waitKey(0)

    def suavidade2():
        imagem = cv2.imread(arquivo)
        loading()

        imagem = imagem[::2,::2] # Diminui a imagem
        suave = np.vstack([
        np.hstack([imagem,
        cv2.GaussianBlur(imagem, ( 3, 3), 0)]), 
        np.hstack([cv2.GaussianBlur(imagem, ( 5, 5), 0), 
        cv2.GaussianBlur(imagem, ( 7, 7), 0)]), 
        np.hstack([cv2.GaussianBlur(imagem, ( 9, 9), 0), 
        cv2.GaussianBlur(imagem, (11, 11), 0)]), 
        ])
        cv2.imshow("Imagem original e suavisadas pelo filtro Gaussiano", suave)
        cv2.waitKey(0)
    
    def suavidade3():
        imagem = cv2.imread(arquivo)
        loading()

        imagem = imagem[::2,::2] # Diminui a imagem
        suave = np.vstack([
        np.hstack([imagem,
        cv2.medianBlur(imagem, 3)]), 
        np.hstack([cv2.medianBlur(imagem, 5), 
        cv2.medianBlur(imagem, 7)]), 
        np.hstack([cv2.medianBlur(imagem, 9), 
        cv2.medianBlur(imagem, 11)]), 
        ])
        cv2.imshow("Imagem original e suavisadas pela mediana", suave)
        cv2.waitKey(0)

    def suavidade4():
        imagem = cv2.imread(arquivo)
        loading()

        imagem = imagem[::2,::2] 
        suave = np.vstack([
            np.hstack([imagem,cv2.bilateralFilter(imagem, 3, 21, 21)]), 
            np.hstack([cv2.bilateralFilter(imagem, 5, 35, 35), cv2.bilateralFilter(imagem, 7, 49, 49)]), 
            np.hstack([cv2.bilateralFilter(imagem, 9, 63, 63), cv2.bilateralFilter(imagem, 11, 77, 77)]) 
        ])
        cv2.imshow("Suavização com filtro bilateral", suave)
        cv2.waitKey(0)

    principal.oghl.clicked.connect(trocaCor)
    principal.canaisImage.clicked.connect(canaisImage)
    principal.rgb.clicked.connect(rgb)
    principal.bgr.clicked.connect(bgr)
    principal.grafico.clicked.connect(grafico1)
    principal.grafico_2.clicked.connect(grafico2)
    principal.suavidade.clicked.connect(suavidade1)
    principal.suavidade_2.clicked.connect(suavidade2)
    principal.suavidade_3.clicked.connect(suavidade3)
    principal.suavidade_4.clicked.connect(suavidade4)

    # FIM MODULO 4

    # INICIO MODULO 5

    def binarizacao1():
        imagem = cv2.imread(arquivo)
        loading()

        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) 

        suave = cv2.GaussianBlur(imagem, (7, 7), 0) 
        (T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)
        (T, binI) = cv2.threshold(suave, 160, 255, 
        cv2.THRESH_BINARY_INV)
        
        resultado = np.vstack([
        np.hstack([suave, bin]),
        np.hstack([binI, cv2.bitwise_and(imagem, imagem, mask = binI)])
        ]) 
        cv2.imshow("Binarização da imagem", resultado)
        cv2.waitKey(0)

    def binarizacao2():
        imagem = cv2.imread(arquivo)
        loading()

        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # converte 
        
        suave = cv2.GaussianBlur(imagem, (7, 7), 0) # aplica blur 
        bin1 = cv2.adaptiveThreshold(suave, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        bin2 = cv2.adaptiveThreshold(suave, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 
        21, 5)
        resultado = np.vstack([
        np.hstack([imagem, suave]),
        np.hstack([bin1, bin2])
        ]) 
        cv2.imshow("Binarização adaptativa da imagem", resultado)
        cv2.waitKey(0)

    def binarizacao3():
        imagem = cv2.imread(arquivo)
        loading()
        
        img = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # converte 
        suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur 
        T = mahotas.thresholding.otsu(suave)
        temp = img.copy()
        temp[temp > T] = 255
        temp[temp < 255] = 0
        temp = cv2.bitwise_not(temp)
        T = mahotas.thresholding.rc(suave)
        temp2 = img.copy()
        temp2[temp2 > T] = 255
        temp2[temp2 < 255] = 0
        temp2 = cv2.bitwise_not(temp2)
        resultado = np.vstack([
        np.hstack([img, suave]),
        np.hstack([temp, temp2])
        ]) 
        cv2.imshow("Binarização com método Otsu e Riddler-Calvard", 
        resultado)
        cv2.waitKey(0)

    def sobel():
        imagem = cv2.imread(arquivo)
        loading()

        img = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        sobel = cv2.bitwise_or(sobelX, sobelY)
        resultado = np.vstack([
        np.hstack([img, sobelX]),
        np.hstack([sobelY, sobel])
        ]) 
        cv2.imshow("Sobel", resultado)
        cv2.waitKey(0)

    def sobel2():
        imagem = cv2.imread(arquivo)
        loading()

        img = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(img, cv2.CV_64F)
        lap = np.uint8(np.absolute(lap))
        resultado = np.vstack([img, lap]) 
        cv2.imshow("Filtro Laplaciano", resultado)
        cv2.waitKey(0)

    def sobel3():
        imagem = cv2.imread(arquivo)

        img = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        suave = cv2.GaussianBlur(img, (7, 7), 0)
        canny1 = cv2.Canny(suave, 20, 120)
        canny2 = cv2.Canny(suave, 70, 200)
        resultado = np.vstack([
        np.hstack([img, suave ]),
        np.hstack([canny1, canny2])
        ]) 
        cv2.imshow("Detector de Bordas Canny", resultado)
        cv2.waitKey(0)

    principal.binarizacao.clicked.connect(binarizacao1)
    principal.binarizacao_2.clicked.connect(binarizacao2)
    # principal.binarizacao_3.clicked.connect(binarizacao3)
    principal.sobel.clicked.connect(sobel)
    principal.sobel_2.clicked.connect(sobel2)
    principal.sobel_3.clicked.connect(sobel3)

    # FIM MODULO 5

    # INICIO MODULO 6

    def FuncaoObjetos():
        img = cv2.imread(arquivo)
        loading()

        def escreve(img, texto, cor=(255,0,0)): #Função para facilitar a escrita nas imagem
            fonte = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, texto, (10,20), fonte, 0.5, cor, 0, cv2.LINE_AA)
            
        imgColorida = cv2.imread(arquivo) #Carregamento da imagem
        img = cv2.cvtColor(imgColorida, cv2.COLOR_BGR2GRAY, 255, 0)
        suave = cv2.blur(img, (7, 7)) #Passo 2: Blur/Suavização da imagem
        T = mahotas.thresholding.otsu(suave) #Passo 3: Binarização resultando em pixels brancos e pretos
        bin = suave.copy()
        bin[bin > T] = 255
        bin[bin < 255] = 0
        bin = cv2.bitwise_not(bin)
        bordas = cv2.Canny(bin, 70, 150) #Passo 4: Detecção de bordas com Canny
        #Passo 5: Identificação e contagem dos contornos da imagem
        #cv2.RETR_EXTERNAL = conta apenas os contornos externos
        contours, hierarchy = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #A variável lx (lixo) recebe dados que não são utilizados
        print(img, "Imagem em tons de cinza", 0)
        print(suave, "Suavização com Blur", 0)
        print(bin, "Binarização com Método Otsu", 255)
        print(bordas, "Detector de bordas Canny", 255)
        temp = np.vstack([np.hstack([img, suave]), np.hstack([bin, bordas])]) 
        cv2.imshow("Quantidade de objetos: " + str(len(contours)), temp)
        cv2.waitKey(0)
        cv2.imshow("Imagem Original", imgColorida)
        
        cv2.drawContours(imgColorida, contours, -1, (0, 255, 0), 2)
        print(imgColorida, str(len(contours)) + "objetos encontrados!")
        principal.qtdObjetos.setText(str(len(contours)))

        cv2.imshow("Resultado", imgColorida)
        cv2.waitKey(0)

    principal.objetos.clicked.connect(FuncaoObjetos)
    # FIM MODULO 6
 


def fecharSistemaPrincipal():
    exit()

def fecharSistemaEquipe():
    equipe.hide()
    app.exec() 

def creditos():
    equipe.show()
    app.exec()   

def fecharAlert():
     principal.textCarregar.setHidden(True)


app=QtWidgets.QApplication([])
principal=uic.loadUi("principal.ui")
equipe=uic.loadUi("equipe.ui")
principal.carregaImage.clicked.connect(carregaImage)
principal.fecharSistema.clicked.connect(fecharSistemaPrincipal)
principal.creditos.clicked.connect(creditos)
equipe.fecharSistemaEquipe.clicked.connect(fecharSistemaEquipe)
principal.progressBar.setHidden(True)

principal.show()
app.exec()





