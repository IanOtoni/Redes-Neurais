import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1

base = pd.read_csv('autos.csv', encoding='ISO-8859-1')
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)
base = base.drop('name', axis = 1)
base = base.drop('seller', axis = 1)
base = base.drop('offerType', axis = 1)

base['price'] = pd.to_numeric(base['price'], errors='coerce') #Convertendo a coluna 'price' para numérica e tratando os erros
base = base.dropna(subset=['price']) # Removendo as linhas onde 'price' é NaN
base['price'] = base['price'].astype(int) # Convertendo 'price' de float para int
i1 = base.loc[base.price <= 10]
base = base[base.price > 10]
i2 = base.loc[base.price > 350000]
base = base[base.price < 350000]

n1 = base.loc[pd.isnull(base['vehicleType'])] #limousine 
n2 = base.loc[pd.isnull(base['gearbox'])] #manuell
n3 = base.loc[pd.isnull(base['model'])] #golf
n4 = base.loc[pd.isnull(base['fuelType'])] #benzin
n5 = base.loc[pd.isnull(base['notRepairedDamage'])] #nein

valores = {'vehicleType': 'limousine', 'gearbox': 'manuell', 'model': 'golf', 
           'fuelType': 'benzin', 'notRepairedDamage': 'nein'}
base = base.fillna(value = valores) #recurso do pandas


previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0]) #transforma classe de string para numero
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])


onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough') #atualizado
previsores = onehotencoder.fit_transform(previsores).toarray() #atualizado

regressor = Sequential([
    tf.keras.layers.Dense(units=158, activation='relu', input_dim=316),                  
    tf.keras.layers.Dense(units=158, activation='relu'), 
    tf.keras.layers.Dense(units=1, activation='linear') #linear n faz nada e é melhor para regressão
])
regressor.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)

previsoes = regressor.predict(previsores)
resultado = regressor.evaluate(previsores, preco_real)

media_resposta = preco_real.mean()
media_calculada = previsoes.mean()


