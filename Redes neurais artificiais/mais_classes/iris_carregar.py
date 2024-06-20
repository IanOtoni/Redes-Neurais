import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json # atualizado: tensorflow==2.0.0-beta1

arquivo = open('classificador_iris.json')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_iris.weights.h5')

teste = np.array([[6.3,3.3,4.7,1.6]])
previsao = classificador.predict(teste)


if(previsao[0][0]>0.51).any():
    print('setosa')
    
elif(previsao[0][1]>0.51).any():
    print('versicolor')
    
elif(previsao[0][2]>0.51).any():
    print('virginica')