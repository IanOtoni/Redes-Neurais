import pandas as pd

previsores = pd.read_csv('entradas_breast.csv')
classe =pd.read_csv('saidas_breast.csv')

#Divisão entre treinamento e teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25)

import keras
from keras.models import Sequential #modelo sequencial de rede neural
from keras.layers import Dense #Camadas densas na rede neural(cada neuronio ligado em todos os outros da prox camada)

classificador = Sequential() #criando rede neural
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30)) #Adicionar primeira camada oculta
#Primeira camada oculta e definicao da camada de entrada ja foi
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))

classificador.add(Dense(units = 1, activation = 'sigmoid')) #camaada de saida

otimizador = keras.optimizers.Adam(learning_rate = 0.001, decay = 0.0001, clipvalue = 0.5)
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)


pesos0 = classificador.layers[0].get_weights()
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)