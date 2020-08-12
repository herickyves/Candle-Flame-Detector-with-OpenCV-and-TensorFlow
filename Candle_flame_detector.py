#----------------------------------------------------------------------------------------
# Developed by Herick Yves Silva Ribeiro.
# Instituto Politécnico da Guarda, Guarda, Portugal, 2020. 
# e-mail: herick.yves@hotmail.com
#----------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
#                               Libraries
#--------------------------------------------------------------------------------
import cv2          #Biblioteca de visão computacional: OpenCV.
import numpy as np  #Biblioteca para manipulação de matrizes: Numpy.
import math         #Biblioteca de funções matematicas.
import heapq
import time
import Classificador as cl

#--------------------------------------------------------------------------------
#                               Variables
#--------------------------------------------------------------------------------
lista = []
imgv = []
idi = 0

#--------------------------------------------------------------------------------
#                               Constants
#--------------------------------------------------------------------------------
color_vetor = [(0,0,255),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
PER_SIZES = 0.8
MIN_SIZE = 10
RE_SIZE_W = 800
RE_SIZE_H = 600
MIN_CONTOURS = 30
MIN_THRES = 250
MAX_THRES = 255
VIDEO_NAME = 'video2'
URL = "C:\\Users\\heric\\Music\\redes neurais 2\\" + VIDEO_NAME + ".mp4"

#--------------------------------------------------------------------------------
#                         Neural Network Setup
#--------------------------------------------------------------------------------
cla = cl.Classificador()
cla.carregar_modelo()

#--------------------------------------------------------------------------------
#                         Image capture Mode
#--------------------------------------------------------------------------------
#Inicia a captura da imagem.
cap = cv2.VideoCapture(URL) 
#cap = cv2.VideoCapture(0)

#--------------------------------------------------------------------------------
#                         Functions
#--------------------------------------------------------------------------------

# fix_coordinates it's a function to centralize the region of interest, before classify it. 
def fix_coordinates(x,y,w,h,per_Size):
        per_size = per_Size
        if(x-int(per_size*w)< 0):
                x1 = 0
        else:
                x1 = x-int(per_size*w)

        if(y-int(per_size*h) < 0):
                y1 = 0
        else:
                y1 = y-int(per_size*h)
        if(x+int(w*(1+per_size))> img2.shape[0]):
                w1 = img2.shape[0]
        else:
                w1 = x+int(w*(1+per_size))
        
        if( y+int(h*(1+per_size)) > img2.shape[1] ): 
                h1 = img2.shape[1]
        else:
                h1 = y+int(h*(1+per_size))
        return x1,y1,w1,h1

#--------------------------------------------------------------------------------
#                         Main Loop
#--------------------------------------------------------------------------------
#Loop para capturar frame a frame do video.
while(cap.isOpened()):
    
        #Busca um frame. 
        _,img = cap.read()

        #Verifica se este frame contém alguma informação
        if img is not None:

                #Diminui o tamanho da imagem para facilitar o processamento
                img = cv2.resize(img, (RE_SIZE_W, RE_SIZE_H), interpolation = cv2.INTER_CUBIC)
                
                #Cria copias da imagem original para uso posterior
                img4 = img.copy()
                img2 = img.copy()
                img5 = img.copy()
                for i in range(5):
                        imgv.append(img.copy())
                               
                cv2.imshow("Original",img)
                
                #Transforma a imagem original em escala de cinza, para facilitar o processamento de imagem.
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                #Mapeia os pixels com valores entre 250 e 255, e transforma eles no valor 1 e o restante coloca o valor 0.
                _,img = cv2.threshold(img,MIN_THRES,MAX_THRES,cv2.THRESH_BINARY)

                #Faz copias da imagem ja binarizada para uso posterior.
                thres = img

                #Encontra contornos na imagem mapeada e binarizada.
                _,contours,_ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                tamanho = []
                maior = []

                #Organiza por tamanho dos contornos
                for (i, c) in enumerate(contours):
                        tamanho.append((len(c),i))
                        
                # verifica se a lista de tamanho esta vazia       
                if(len(tamanho) != 0):
                        maxi = max(tamanho) # maxi[1] é o indice do maior contorno, maxi[0] é tamanho do maior contorno.
                        if(maxi[0] > MIN_CONTOURS):
                                if(len(tamanho) == 1 and tamanho[0][0] > MIN_SIZE):
                                        maior.append(maxi)
                                        (x, y, w, h) = cv2.boundingRect(contours[0])
                                        x2,y2,w2,h2 = fix_coordinates(x,y,w,h,PER_SIZES)
                                        imgv[0] = img4[y2:h2, x2:w2]
                                        result = cla.predizer_imagem(imgv[0])
                                        if(result == True):
                                                cv2.rectangle(img2,(x2,y2),(w2,h2),color_vetor[0],2)
                                                #cv2.putText(img2,"Fogo",((int(contours[maior[i]][0][0][0])+2),(int(contours[maior[i]][0][0][1])+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)
                                elif(len(tamanho) >= 2 and len(tamanho) <= 4 ):
                                        maior = heapq.nlargest(2,tamanho)
                                        for i in range(len(maior)):
                                                if(len(contours[maior[i][1]]) > MIN_CONTOURS):
                                                        #cv2.drawContours(img2, contours, maior[i][1] , color_vetor[i], 3)
                                                        (x, y, w, h) = cv2.boundingRect(contours[maior[i][1]])
                                                        x2,y2,w2,h2 = fix_coordinates(x,y,w,h,PER_SIZES)
                                                        imgv[i] = img4[y2:h2, x2:w2]
                                                        result = cla.predizer_imagem(imgv[i])
                                                        if(result == True):
                                                                cv2.rectangle(img2,(x2,y2),(w2,h2),color_vetor[0],2)
                                                                #cv2.putText(img2,"Fogo",((int(contours[maior[i]][0][0][0])+2),(int(contours[maior[i]][0][0][1])+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)        
                                elif(len(tamanho) >= 5):
                                        maior = heapq.nlargest(5,tamanho)
                                        for i in range(len(maior)):
                                                if(len(contours[maior[i][1]]) > MIN_CONTOURS):
                                                        #cv2.drawContours(img2, contours, maior[i][1] , color_vetor[i], 3)
                                                        (x, y, w, h) = cv2.boundingRect(contours[maior[i][1]])
                                                        x2,y2,w2,h2 = fix_coordinates(x,y,w,h,PER_SIZES)
                                                        imgv[i] = img4[y2:h2, x2:w2]
                                                        result = cla.predizer_imagem(imgv[i])
                                                        if(result == True):
                                                                cv2.rectangle(img2,(x2,y2),(w2,h2),color_vetor[0],2)
                                                                #cv2.putText(img2,"Fogo",((int(contours[maior[i]][0][0][0])+2),(int(contours[maior[i]][0][0][1])+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)
              
                #Mostra os resultados na tela.
                cv2.imshow("threshold",thres)
                cv2.imshow("Imagem com os contornos",img2)

                #Verifica se a tecla 'Esc' foi preciosada. Em caso afirmativo sai do loop de captura de imagem.
                k = cv2.waitKey(5) & 0xFF
                if k == 27 or cap.isOpened() != True:
                    break

#Encerra a captura de imagem desligando a camera.
cap.release()

#Fecha todas as janelas.
cv2.destroyAllWindows()



###################################################################################################################################################################################################################################
# L

'''blur = cv2.GaussianBlur(img,(5,5),0)
                #cv2.imshow("blur",blur)
                canny = cv2.Canny(blur,30,150)
                cv2.imshow("Canny",canny)
                lines = cv2.HoughLinesP(canny,2,np.pi/180,100,np.array([]),minLineLength = 20,maxLineGap=20)

                line_image = np.zeros_like(canny)
                if lines is not None:
                        for line in lines:
                                x1,y1,x2,y2 = line.reshape(4)
                                cv2.line(img5,(x1,y1),(x2,y2), (255,0,0),3)

                #cv2.imshow("lines",img2)'''


###################################################################################################################################################################################################################################

'''cv2.drawContours(img2, contours, maxi[1] , (0, 255, 0), 3)
                                #cv2.putText(img2,"Flame Position:"+str(contours[maxi[1]][0][0]),((int(contours[maxi[1]][0][0][0])+2),(int(contours[maxi[1]][0][0][1])+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)
                                
                                #Cria um retangulo tangente as bordas do contorno. Esta função retorna a posição dos dois vertices diagonais do retangulo.
                                (x, y, w, h) = cv2.boundingRect(contours[maxi[1]])
                                x2,y2,w2,h2 = fix_coordinates(x,y,w,h,0.6)
                                img4 = img4[y2:h2, x2:w2]
                                
                                if(img4 is not None):
                                        print(str(img4.shape))
                                        result = cla.predizer_imagem(img4)
                                        if(result == True):
                                                if(h<w):
                                                        cv2.rectangle(img2,(x,y),(x+w2,y+w2),(0,255,0),2)
                                                        cv2.putText(img2,"Fogo",((int(contours[maxi[1]][0][0][0])+2),(int(contours[maxi[1]][0][0][1])+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)
                                                else:
                                                        cv2.rectangle(img2,(x,y),(x+h2,y+h2),(0,255,0),2)
                                                        cv2.putText(img2,"Fogo",((int(contours[maxi[1]][0][0][0])+2),(int(contours[maxi[1]][0][0][1])+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)
                                        else:
                                                if(h<w):
                                                        cv2.rectangle(img2,(x,y),(x+w2,y+w2),(0,0,255),2)
                                                        cv2.putText(img2,"Nao e fogo",((int(contours[maxi[1]][0][0][0])+2),(int(contours[maxi[1]][0][0][1])+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)
                                                else:
                                                        cv2.rectangle(img2,(x,y),(x+h2,y+h2),(0,0,255),2)
                                                        cv2.putText(img2,"Nao e Fogo",((int(contours[maxi[1]][0][0][0])+2),(int(contours[maxi[1]][0][0][1])+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)'''
                                #cv2.rectangle(img2,(x2,y2),(w2,h2),(0,255,0),2)
                                #print('('+str(x1)+','+str(y1)+')'+' ('+str(x+w)+','+str(y+h)+')')
