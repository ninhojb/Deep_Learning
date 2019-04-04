import numpy as np
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense, Dropout , LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


base = pd.read_csv('petr4-treinamento.csv')
base = base.dropna()
base_treinamento = base.iloc[:,1:2].values
base_valor_maximo = base.iloc[:, 2:3].values

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizado = normalizador.fit_transform(base_treinamento)
base_valor_maximo_normalizada = normalizador.fit_transform(base_valor_maximo)

previsores = []
preco_real1 = []
preco_real2 = []

for i in range(90, 1242):
    previsores.append(base_treinamento_normalizado[i-90:i,0])
    preco_real1.append(base_treinamento_normalizado[i, 0])
    preco_real2.append(base_valor_maximo_normalizada[i, 0])
previsores, preco_real1, preco_real2  = np.array(previsores), np.array(preco_real1), np.array(preco_real2)
previsores = np.reshape(previsores,(previsores.shape[0], previsores.shape[1], 1))

preco_real = np.column_stack((preco_real1,preco_real2))

regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(previsores.shape[1], 1)))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))
regressor.add(Dense(units=2, activation='linear'))
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

regressor.fit(previsores, preco_real, batch_size=32, epochs=100)

base_teste = pd.read_csv('petr4-teste.csv')
preco_real_open = base_teste.iloc[:,1:2].values
preco_real_high = base_teste.iloc[:,2:3].values

base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)
entradas = base_completa[len(base_completa)- len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

x_teste = []
for i in range(90,112):
    x_teste.append(entradas[i-90:i , 0])
x_teste = np.array(x_teste)
x_teste = np.reshape(x_teste,(x_teste.shape[0], x_teste.shape[1], 1))

previsoes = regressor.predict(x_teste)
previsoes = normalizador.inverse_transform(previsoes)

plt.plot(preco_real_open, color='red', label='Preço abertura Real')
plt.plot(preco_real_high, color='black', label='Preço alta Real')
plt.plot(previsoes[:,0], color='blue', label='Previsao abertura')
plt.plot(previsoes[:,1], color='orange', label='Previsao alta')
plt.title('Previsão de preço das açoes')
plt.xlabel('Tempo')
plt.ylabel('Valor yahoo')
plt.legend()
plt.show
