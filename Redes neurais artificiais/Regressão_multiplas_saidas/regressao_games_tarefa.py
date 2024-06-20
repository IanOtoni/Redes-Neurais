import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

base = base.dropna(axis = 0)
base = base.loc[base['NA_Sales'] > 1] #Tira as linhas que a vena na América do Norte foi menor que 1
base = base.loc[base['EU_Sales'] > 1] #Agora para Europa

base['Name'].value_counts()
base = base.drop('Name', axis = 1)

previsores = base.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]].values
vendas_totais = base.iloc[:, 7].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 11] = labelencoder.fit_transform(previsores[:, 8])

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,11])], remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray() #atualizado

camada_entrada = Input(shape=(152,))
camada_oculta1 = Dense(units = 77, activation='sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 77, activation='sigmoid')(camada_oculta1)
camada_saida = Dense(units = 1, activation='linear')(camada_oculta2)

regressor = Model(inputs = camada_entrada, outputs = camada_saida)

regressor.compile(optimizer='adam', loss='mse')
regressor.fit(previsores, vendas_totais, batch_size = 100, epochs = 5000)
previsao_mundo = regressor.predict(previsores)
