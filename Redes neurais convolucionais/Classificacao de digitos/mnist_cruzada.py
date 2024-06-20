import numpy as np
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from keras.datasets import mnist
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

seed = 5
np.random.seed(seed)

(X, y), (X_teste, Y_teste) = mnist.load_data()
previsores = X.reshape(X.shape[0], 28, 28, 1)
previsores = previsores.astype('float32')
previsores /= 255
classe = to_categorical(y, 10)

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
resultados = []

a = np.zeros(5)
b = np.zeros((classe.shape[0], 1))

for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros((classe.shape[0], 1))):
    
    classificador = Sequential([ #Primeira camada é o operador de convolução
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(pool_size = (2, 2)), 
        tf.keras.layers.Flatten(), #flatening
        tf.keras.layers.Dense(units=128, activation = 'relu'),
        tf.keras.layers.Dense(units=10, activation = 'softmax')
        ])

    classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento], 
                      batch_size = 128, epochs = 5)
    
    precisao = classificador.evaluate(previsores[indice_teste], classe[indice_teste])
    resultados.append(precisao[1])
print(sum(resultados) / len(resultados))