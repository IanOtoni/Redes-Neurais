import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.utils import to_categorical

base = pd.read_csv('iris.csv') #Le a base de dados
previsores = base.iloc[:, 0:4].values #separa os previsores
classe = base.iloc[:, 4].values #separa as classes
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe) #transforma classe de string para numero
classe_dummy = to_categorical(classe) #faz com que tenha 3 saídas


classificador = Sequential([
    tf.keras.layers.Dense(units=4, activation='relu', input_dim=4),                  
    tf.keras.layers.Dense(units=8, activation='softmax'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=3, activation='softmax')
])
classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
classificador.fit(previsores, classe_dummy, batch_size = 10, epochs = 1000) #treina


classificador_json = classificador.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)
    
classificador.save_weights('classificador_iris.weights.h5')