import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

base = base.dropna(axis = 0)
base = base.loc[base['NA_Sales'] > 1] #Tira as linhas que a vena na América do Norte foi menor que 1
base = base.loc[base['EU_Sales'] > 1] ##Agora para Europa

base['Name'].value_counts()
nome_jogos = base['Name']
base = base.drop('Name', axis = 1)

previsores = base.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])], remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray() #atualizado

camada_entrada = Input(shape=(61,))
camada_oculta1 = Dense(units = 32, activation='sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 32, activation='sigmoid')(camada_oculta1)
camada_saida1 = Dense(units = 1, activation='linear')(camada_oculta2)
camada_saida2 = Dense(units = 1, activation='linear')(camada_oculta2)
camada_saida3 = Dense(units = 1, activation='linear')(camada_oculta2)

regressor = Model(inputs = camada_entrada, 
                  outputs = [camada_saida1, camada_saida2, camada_saida3])

regressor.compile(optimizer='adam', loss='mse')
regressor.fit(previsores, [venda_na, venda_eu, venda_jp], batch_size = 100, epochs = 5000)
previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)


