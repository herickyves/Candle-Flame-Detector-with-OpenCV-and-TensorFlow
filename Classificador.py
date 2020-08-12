#--------------------------------------------------------------------------------
# Developed by Herick Yves Silva Ribeiro.
# Instituto Politécnico da Guarda, Guarda, Portugal, 2020. 
# e-mail: herick.yves@hotmail.com
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
#                               Libraries
#--------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
import random
import time

#--------------------------------------------------------------------------------
#                               Constants
#--------------------------------------------------------------------------------
height = 35
width = 35

#--------------------------------------------------------------------------------
#                               Functions
#--------------------------------------------------------------------------------

imagem = cv2.imread('C:\\Users\\heric\\Music\\redes neurais 2\\dataset\\fogo\\img199.jpg')
imagem = cv2.resize(imagem, (width, height), interpolation = cv2.INTER_CUBIC)

def create_model():
        model = keras.Sequential([
        
        keras.layers.InputLayer(imagem.shape),
        keras.layers.Flatten(input_shape=(height,width)),
        keras.layers.Dense(500, activation=tf.nn.relu),
        keras.layers.Dense(150, activation=tf.nn.relu),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
        
        ])
        return model

modelo = create_model()
#--------------------------------------------------------------------------------
#                               Class
#--------------------------------------------------------------------------------
class Classificador(object):

    def __init__(self):
        self.modelo = modelo
    
    def carregar_modelo(self):

        model = create_model()
        model.load_weights("Classificador_de_chama.h5")
        opt = tf.keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=opt, 
                    loss='binary_crossentropy',   
                    metrics=['accuracy'])
        self.modelo = model

    def predizer_imagem(self, imagem):
        if(imagem.size != 0):
            imagem = cv2.resize(imagem, (width, height), interpolation = cv2.INTER_CUBIC)
            imagem_predict = np.expand_dims(imagem, axis=0)
            resultado = self.modelo.predict(imagem_predict)
            if(resultado >= 0.5):
                result = True
            elif(resultado < 0.5):
                result = False  
            return result