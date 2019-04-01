import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np



(X_treinamento, y_treinamento),( X_teste, y_teste) = mnist.load_data()

previsores_treinamneto = X_treinamento.reshape(X_treinamento.shape[0],
                                               28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0],
                                   28,28,1)
previsores_treinamneto = previsores_treinamneto.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamneto /= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

classificardor = Sequential()

classificardor.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))

classificardor.add(MaxPooling2D(pool_size=(2,2)))

classificardor.add(Flatten())

classificardor.add((Dense(units=128, activation='relu')))
classificardor.add(Dense(units=10, activation='softmax'))
classificardor.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classificardor.fit(previsores_treinamneto,classe_treinamento,
                   batch_size=128, epochs=5,
                   validation_data=(previsores_teste, classe_teste))

resultados = classificardor.evaluate(previsores_teste,classe_teste)
print(resultados)

plt.imshow(X_teste[10], cmap = 'gray')
plt.title('Classe ' + str(y_teste[10]))
plt.show()

imagem_teste = X_teste[10].reshape( 1,28,28,1)
imagem_teste = imagem_teste.astype('float32')
imagem_teste /=255
previsao = classificardor.predict(imagem_teste)

print(np.argmax(previsao))
