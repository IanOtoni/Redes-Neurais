import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from tensorflow.keras import backend as k 

base = pd.read_csv('iris.csv') #Le a base de dados
previsores = base.iloc[:, 0:4].values #separa os previsores
classe = base.iloc[:, 4].values #separa as classes
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe) #transforma classe de string para numero
classe_dummy = to_categorical(classe) #faz com que tenha 3 saídas

def criar_rede():
    k.clear_session()
    classificador = Sequential([tf.keras.layers.Dense(units=4, activation='relu', input_dim=4),                  
        tf.keras.layers.Dense(units=8, activation='softmax'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=3, activation='softmax')])
    classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy']) #compila
    return classificador

classificador = KerasClassifier(model = criar_rede,
                                epochs = 1000,
                                batch_size = 10)

resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe_dummy,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()