import os.path
import sys
import numpy as np
import pandas as pd
import csv
import time
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

#INICIO DEL PROGRAMA
print("EVALUACIÓN DEL CLASIFICADOR - " + sys.argv[0] )




#DECLARACIÓN CONSTANTES
L_CLF = 'xgboost'




#LECTURA DE ARGUMENTOS
num_caract = sys.argv[1]
selector = sys.argv[2]
print("ARGUMENTOS: num_caract=" + str(num_caract) + " selector=" + str(selector))




#APERTURA FICHERO
print ("LECTURA DE FICHERO")
X_train = np.loadtxt("./datos/features" + str(num_caract) + "selector" + str(selector) + "/X_train.csv", delimiter=',')
y_train = np.loadtxt("./datos/features" + str(num_caract) + "selector" + str(selector) + "/y_train.csv", delimiter=',')
X_test = np.loadtxt("./datos/features" + str(num_caract) + "selector" + str(selector) + "/X_test.csv", delimiter=',')
y_test = np.loadtxt("./datos/features" + str(num_caract) + "selector" + str(selector) + "/y_test.csv", delimiter=',')
le = pickle.load(open('./modelos/le.sav', 'rb'))

# Configurar y entrenar el modelo XGBoost
model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, seed=42)
model.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo XGBoost:", accuracy)
