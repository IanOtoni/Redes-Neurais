import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data() #Divide a databse
#plt.imshow(X_treinamento[109], cmap = 'grey')
#plt.title('Classe ' + str(y_treinamento[0]))


previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32') #Deixar float para dividir
previsores_treinamento /= 255 #Deixar entre 0 e 1
previsores_teste /= 255

classe_treinamento = to_categorical(y_treinamento, 10) #faz com que tenha 10 saídas
classe_teste = to_categorical(y_teste, 10)

classificador = Sequential([ #Primeira camada é o operador de convolução
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation = 'relu'), #numero mapas, kernel_size, dimensões da imagem
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)), #Tamanho da matriz q pega o valor max, Pooling
    
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'), #Mais uma camada de convolução
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
    
    tf.keras.layers.Flatten(), #flatening
    
    tf.keras.layers.Dense(units=128, activation = 'relu'), #rede neural densa agora
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=128, activation = 'relu'),   
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=10, activation = 'softmax')
    ])

classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 128, epochs = 5) #,validation_data = (previsores_teste, classe_teste))

resultado = classificador.evaluate(previsores_teste, classe_teste)

