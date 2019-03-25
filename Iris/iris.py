import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('iris.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
labelencoder = LabelEncoder()
'''
iris setosa     1 0 0
iris virginica  0 1 0
iris versicolor 0 0 1
'''
classe = labelencoder.fit_transform(classe)

classificador = Sequential()
classificador.add(Dense(units=4, activation='tanh',kernel_initializer='normal', input_dim=4))

classificador.add(Dropout(0.3))
classificador.add(Dense(units=4, activation='tanh',kernel_initializer='normal'))

classificador.add(Dropout(0.3))
classificador.add(Dense(units=3, activation='softmax'))

classificador.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

classificador.fit(previsores,classe,
                  batch_size=10, epochs=1000,
                  )
planta = np.array([[4.5,2,3.4,1.2]])

previsao = classificador.predict(planta)

print(previsao > 0.5)

if previsao[:,:1]>0.5:
    print('Resultado: {}\nÉ uma: iris setosa'.format(previsao))
elif previsao[:,1:2]>0.5:
    print('Resultado: {}\nÉ uma: iris virginica'.format(previsao))
else:
    print('Resultado: {}\nÉ uma: iris versicolor'.format(previsao))
