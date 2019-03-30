import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils


(X_treinamento, y_treinamento),( X_teste, y_teste) = mnist.load_data()
plt.imshow(X_treinamento[0], cmap='gray')# fica de sinza
plt.title('Classe' + str(y_treinamento[0]))

#formato do tensoflow
previsores_treinamneto = X_treinamento.reshape(X_treinamento.shape[0],
                                               28, 28, 1)

previsores_teste = X_teste.reshape(X_teste.shape[0],
                                   28,28,1)

previsores_treinamneto = previsores_treinamneto.astype('float32')
previsores_teste = previsores_teste.astype('float32')

#normalizaçao intervalo entre 0 e 1
previsores_treinamneto /= 255
previsores_teste /= 255

#criar as classe do tipo dummy
# numero 0 = 1 0 0 0 0 0 0
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)


classificardor = Sequential()
# etapa 1 - Operador de convulacional detector de map(kernekl) recomendado 64(kernel)
#(3,3)tamanho do detector de caracteristica
classificardor.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
#etapa 2 - Pooling
classificardor.add(MaxPooling2D(pool_size=(2,2)))
#etapa 3 - Flattening
classificardor.add(Flatten())
#etapa 4 - Rede Neural Densa
classificardor.add((Dense(units=128, activation='relu')))
classificardor.add(Dense(units=10, activation='softmax'))
classificardor.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classificardor.fit(previsores_treinamneto,classe_treinamento,
                   batch_size=128, epochs=20,
                   validation_data=(previsores_teste, classe_teste))

resultados = classificardor.evaluate(previsores_teste,classe_teste)
print(resultados)

#consederar o valor do val_acc
#plt.show()
