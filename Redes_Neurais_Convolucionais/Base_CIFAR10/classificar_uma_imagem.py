from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np


(x_treinamento, y_treinamento),(x_teste,y_teste) = cifar10.load_data()

previsores_treinamento = x_treinamento.reshape(x_treinamento.shape[0],32,32,3)

previsores_teste = x_teste.reshape(x_teste.shape[0], 32, 32 ,3)

previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255
previsores_teste /= 255
classe_treinamento = np_utils.to_categorical(y_treinamento,10)
classe_teste = np_utils.to_categorical(y_teste, 10)

#rede convolucionaria
classificardor = Sequential()

classificardor.add(Conv2D(32, (3,3), input_shape=(32,32,3),activation='relu',
                          padding='same',use_bias=True,
                              kernel_initializer='glorot_normal'))
classificardor.add(BatchNormalization())
classificardor.add(MaxPooling2D(2,2))


classificardor.add(Conv2D(32, (3,3), input_shape=(32,32,3),
                          activation='relu' ,padding='same',use_bias=True,
                              kernel_initializer='glorot_normal'))
classificardor.add(BatchNormalization())
classificardor.add(MaxPooling2D(2,2))

classificardor.add(Flatten())

#Rede Neural
classificardor.add((Dense(units=128, activation='relu', use_bias=True)))
classificardor.add(Dropout(0.2))
classificardor.add(Dense(units=128, activation='relu', use_bias=True))
classificardor.add(Dropout(0.2))


classificardor.add(Dense(units=10, activation='softmax'))
classificardor.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#treinamento
classificardor.fit(previsores_treinamento,classe_treinamento,
                   batch_size=128, epochs=5,
                   validation_data=(previsores_teste, classe_teste))

resultados = classificardor.evaluate(previsores_teste,classe_teste)
print(resultados)
print('Media: ',resultados[1])


elementos = ['airplanes','cars', 'birds','cats', 'deer', 'dogs', 'frogs', 'horses', 'ships','trucks']

plt.imshow(x_teste[0], cmap = 'gray')
plt.title('Classe ' + str(y_teste[0]))
plt.show()

imagem_teste = x_teste[0].reshape( 1,32,32,3)
imagem_teste = imagem_teste.astype('float32')
imagem_teste /=255
previsao = classificardor.predict(imagem_teste)

for i in range(len(elementos)):
    if previsao[0][i] >0.3:
        print(elementos[i])

print(previsao)
print(np.argmax(previsao))
