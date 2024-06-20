import numpy as np
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

rede_neural = Sequential([
    Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size = (2, 2)),
    
    Conv2D(64, (3, 3), activation = 'relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size = (2, 2)),
    
    Flatten(),
    
    Dense(units = 128,  activation ='relu'),
    Dropout(0.2),
    Dense(units = 128,  activation ='relu'),
    Dropout(0.2),
    Dense(units = 1,  activation ='sigmoid',)
    ])
rede_neural.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
gerador_treinamento = ImageDataGenerator(rescale = 1./255, 
                                         rotation_range = 7, 
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset_personagens/training_set', 
                                                           target_size = (64, 64),
                                                            batch_size = 32,
                                                            class_mode = 'binary')
base_teste = gerador_teste.flow_from_directory('dataset_personagens/test_set', 
                                                           target_size = (64, 64),
                                                            batch_size = 32,
                                                            class_mode = 'binary')
rede_neural.fit(base_treinamento, steps_per_epoch = 200,
                  epochs = 30, validation_data = base_teste, 
                  validation_steps = 100)
acuracia = rede_neural.evaluate(base_teste)