import sys
import matplotlib.pyplot as plt
import imp
import pandas as pd
from Perceptron import Perceptron

datacv = pd.read_csv("antropo-latinos.csv")

dataaux = datacv.copy()
dataaux['peso'] = datacv['estatura']
dataaux['estatura'] = datacv['peso']
dataaux.columns = ['estatura', 'peso', 'sexo']

data = dataaux.values.tolist()

pr = Perceptron(3,0.1)
weights = []
errores = []

# fase de entrenamiento
for _ in range(100):
    for persona in data:
        # la salida es la variable sexo (ultimo dato de la lista)
        output = persona[-1]
        # bias +x1 + x2 + ...
        inp = [1] + persona[0:-1]
        weights.append(pr.w)
        err = pr.train(inp, output)
        errores.append(err)
        

#valores para evaluar la respuesta del perceptron
estatura = 1.70
peso = 71.0

if pr.predict([1, estatura, peso]) == 1:
    print('Mujer')
else:
    print('Hombre')
    
puede_graficar = True
try:
    imp.find_module('matplotlib')
except:
    puede_graficar= False

if not puede_graficar:
    print('No es posible graficar los resultados porque hace fala matplotlib')
    sys.exit(0)
    pass

plt.plot(errores)
plt.show()