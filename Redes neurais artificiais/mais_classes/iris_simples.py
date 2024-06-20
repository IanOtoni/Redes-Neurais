import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.utils import to_categorical

base = pd.read_csv('iris.csv') #Le a base de dados
previsores = base.iloc[:, 0:4].values #separa os previsores
classe = base.iloc[:, 4].values #separa as classes

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe) #transforma classe de string para numero
classe_dummy = to_categorical(classe) #faz com que tenha 3 saídas
# iris setosa     1 0 0
# iris virginica  0 1 0
# iris versicolor 0 0 1

from sklearn.model_selection import train_test_split #Divide em treinamento e teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

classificador = Sequential() #Cria a rede neural
classificador.add(Dense(units=4, activation = 'relu', input_dim=4))
classificador.add(Dense(units=4, activation = 'relu'))
classificador.add(Dense(units=3, activation = 'softmax'))

classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy']) #compila
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10,
                  epochs = 1000) #treina

resultado = classificador.evaluate(previsores_teste, classe_teste) #ve os resultados
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(previsoes, classe_teste)