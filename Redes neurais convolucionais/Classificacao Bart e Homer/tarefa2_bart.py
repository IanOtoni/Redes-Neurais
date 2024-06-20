import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 

base_de_dados = pd.read_csv('personagens.csv')
previsores = base_de_dados.iloc[:, 0:6].values #separa os previsores
classe = base_de_dados.iloc[:, 6].values #separa as classes

labelencoder = LabelEncoder()
classe_normalizada = labelencoder.fit_transform(classe) #Bart = 0, homer = 1

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_normalizada, test_size=0.20)

rede_neural = Sequential([
    Dense(units = 4, activation = 'relu', input_dim = 6),
    Dense(units = 8, activation = 'relu'),
    Dense(units = 4, activation = 'relu'),
    Dense(units = 1, activation = 'softmax')
    ])
rede_neural.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

rede_neural.fit(previsores_treinamento, classe_treinamento, batch_size = 50, epochs = 1500)

acuracia = rede_neural.evaluate(previsores_teste, classe_teste)