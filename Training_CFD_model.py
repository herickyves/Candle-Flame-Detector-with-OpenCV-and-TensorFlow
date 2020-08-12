#--------------------------------------------------------------------------------
# Instituto Politécnico da Guarda, Guarda, Portugal, 2020. 
# e-mail: herick.yves@hotmail.com
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
#                                        Libraries
#--------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

#--------------------------------------------------------------------------------
#                               Variables
#--------------------------------------------------------------------------------
#Tamanho da imagem de treino.
height = 35
width = 35
image = []
label = []
i = 0
#Da nome as labels para ser usado posteriormente
class_names = ['Acesa','Apagada']

#---------------------------------------------------------------------------------
#                                  TensorFlow Setup
#---------------------------------------------------------------------------------
# Configuração da placa de video.
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config) 
keras.backend.set_session(sess) 



#Importa as imagens.
imagePaths = sorted(list(paths.list_images("C:\\Users\\heric\\Music\\Novo data set\\dataset\\")))

#Mistura o dataSet
random.seed(42)
random.shuffle(imagePaths)

print("Import images begin...")

#Transforma as iamgens em um np.array para ser criado o dataset
for imagePath in imagePaths:
    print(str(i+1)+"/"+str(len(imagePaths)))
    img = cv2.imread(imagePath)
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
    img = np.array(img[:,:,:3])
    if img[0][0].shape[0] < 3:
        continue
    image.append(img)
    poke = imagePath.split(os.path.sep)[-2]
    label.append(poke)
    i+=1
print("Import images Finish!")


#Separa as caracteristicas do data-set em image sendo a Imagem e label sendo sua Label(Classificação "aceso ou apagado")
image = np.array(image, dtype="float") / 255.0
label = np.array(label)


print("[INFO] Formato da matriz:", image.shape)
print("[INFO] Tamanho da matriz: {:.2f}MB".format(image.nbytes / (1024 * 1000.0)))

print('Tamanho do conjunto (dados):', len(image))
print('Tamanho do conjunto (rótulos):', len(label))
#Transforma as labels em 0 e 1
lb = LabelBinarizer()
label = lb.fit_transform(label)

#separa as amostras em treino e teste.
image_train, image_test, label_train, label_test = train_test_split(image, label, test_size=0.2, random_state=42)

print('image_train:', image_train.shape)
print('image_test:', image_test.shape)
print('label_train:', label_train.shape)
print('label_test:', label_test.shape)

#Gera imagens para melhorar o treino
aug = keras.preprocessing.image.ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")



#Cria a arquitetura da rede.
model = keras.Sequential([
    
    keras.layers.InputLayer(image_train[0].shape),
    keras.layers.Flatten(input_shape=(height,width)),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(150, activation=tf.nn.relu),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
   
])

#Otimiza e compila o modelo.
opt = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, 
              loss='binary_crossentropy',   
              metrics=['accuracy'])
#Treina o modelo.
model.fit(image_train, label_train, epochs=5, shuffle=True, verbose=True, validation_data=(image_test, label_test))

#Avalia o modelo
test_loss, test_acc = model.evaluate(image_test  , label_test)
print('Test accuracy:', test_acc)
imagem = cv2.imread('C:\\Users\\heric\\Music\\redes neurais 2\\dataset\\apagado\\img6.jpg')

#imagem = cv2.imread('C:\\Users\\heric\\Music\\redes neurais 2\\dataset\\fogo\\img324.jpg')
imagem = cv2.resize(imagem, (width, height), interpolation = cv2.INTER_CUBIC)

#Tenta predizer uma imagem.
imagem_predict = np.expand_dims(imagem, axis=0)
resultado = model.predict(imagem_predict)

#Salva o modelo.
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('classificador_de_chama.h5')


