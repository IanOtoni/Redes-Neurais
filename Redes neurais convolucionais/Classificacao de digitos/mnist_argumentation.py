import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from keras.datasets import mnist
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')
previsores_treinamento /= 255 #Deixar entre 0 e 1
previsores_teste /= 255
classe_treinamento = to_categorical(y_treinamento, 10)
classe_teste = to_categorical(y_teste, 10)


classificador = Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation = 'relu'), 
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)), 
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(units=128, activation = 'relu'),
    tf.keras.layers.Dense(units=10, activation = 'softmax')
    ])
classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

gerador_treinamento = ImageDataGenerator(rotation_range = 7,#rotação na imagem
                                         horizontal_flip = True, #Giros horizontais
                                         shear_range = 0.2, #Altera valor dos pixels
                                         height_shift_range = 0.07, #Modificação na altura
                                         zoom_range = 0.2) #Muda o zoom
gerador_teste = ImageDataGenerator()

base_treinamento = gerador_treinamento.flow(previsores_treinamento, classe_treinamento, 
                                            batch_size = 128)
base_teste = gerador_teste.flow(previsores_teste, classe_teste, batch_size = 128)

classificador.fit(base_treinamento, steps_per_epoch = int(600000 / 128), #numeros de etapas a serem gerada antes de ser iniciada uma nova epoca
                            epochs = 5, validation_data = base_teste,
                            validation_steps = int(10000/128))