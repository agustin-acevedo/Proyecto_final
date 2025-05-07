
import os.path
import sys
import numpy as np
import pandas as pd
import csv
import time
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier


#DECLARACIÓN CONSTANTES
L_CLF = 'RF'




#LECTURA DE ARGUMENTOS
num_caract = 5
selector = 'RFE'
print("ARGUMENTOS: num_caract=" + str(num_caract) + " selector=" + str(selector))




#APERTURA FICHERO
print ("LECTURA DE FICHERO")
X_train = np.loadtxt("./datos/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/X_train.csv", delimiter=',')
y_train = np.loadtxt("./datos/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/y_train.csv", delimiter=',')
X_test = np.loadtxt("./datos/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/X_test.csv", delimiter=',')
y_test = np.loadtxt("./datos/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/y_test.csv", delimiter=',')



# 1. Hiperparámetros para Decision Tree
param_dt = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy'],
    'random_state': [5,20,42]
}

# 2. Hiperparámetros para Bagging con árbol base
param_bagging = {
    'n_estimators': [10, 50, 100],
    'max_samples': [0.5, 0.8, 1.0],
    'max_features': [0.5, 0.8, 1.0],
    'estimator__max_depth': [5, 10, None]  # parámetros del árbol base
}

# 3. Hiperparámetros para Random Forest
param_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', None]
}

# Entrenadores base
dt = DecisionTreeClassifier(random_state=42)

# Inicializar el clasificador base
#base_clf = DecisionTreeClassifier()

# Crear el clasificador Bagging
#bagging_clf = BaggingClassifier(base_clf)

#rf = RandomForestClassifier(random_state=42)

# Aplicar GridSearchCV
def buscar_mejores_parametros(nombre_modelo, modelo, param_grid):
    grid = GridSearchCV(modelo, param_grid, cv=5, scoring='f1_macro') #Es más completo porque combina precisión y recall para cada clase → útil si también te importa que no se te escapen ejemplos (falsos negativos).
    grid.fit(X_train, y_train)
    print(f"\n🔍 {nombre_modelo} - Mejores parámetros:")
    print(grid.best_params_)
    print(f"Mejor precisión: {grid.best_score_:.4f}")
    return grid.best_estimator_

#f1_macro, es ideal para problemas multiclase con clases desbalanceadas, como UNSW-NB15.

# Ejecutar búsquedas
mejor_dt = buscar_mejores_parametros("Decision Tree", dt, param_dt)
#mejor_bag = buscar_mejores_parametros("Bagging", bagging_clf, param_bagging)
#mejor_rf = buscar_mejores_parametros("Random Forest", rf, param_rf)
