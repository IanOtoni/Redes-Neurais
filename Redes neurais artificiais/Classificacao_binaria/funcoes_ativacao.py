import numpy as np

def stepFunction(soma):
    if(soma>=1):
        return 1
    return 0

def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))

def tahnFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

def reluFunction(soma):
    if(soma>0):
        return soma
    return 0

def linearFunction(soma):
    return soma

def softmaxFunction(x): #Boa para classificação c mais de duas classes, usa na camada de saída
    ex = np.exp(x)
    return ex / ex.sum()

teste = stepFunction(-1)
teste2 = sigmoidFunction(30)
teste3 = tahnFunction(-0.358)
teste4 = reluFunction(4)
teste5 = reluFunction(1000)
valores = [5.0, 2.0, 1.3]
print(softmaxFunction(valores))