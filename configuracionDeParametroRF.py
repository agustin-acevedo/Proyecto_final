from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import codecs
import pickle
import csv
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# print("LECTURA DE FICHERO DE TRAINING")

# # Abrimos el archivo de training de la misma manera
# with codecs.open("./uploads/UNSW_NB15_training-set.csv", "r", encoding="utf-8-sig") as f:
#     reader = csv.reader(f, delimiter=",")
#     raw_data = list(reader)

# np_data = np.asarray(raw_data, dtype=None)

# X = np_data[:, 0:-2]  # Seleccionar todas las columnas menos las dos últimas
# y = np_data[:, -2]    # Seleccionar la penúltima columna (etiqueta como cadena)



# #PREPROCESAMIENTO
# print("PREPROCESAMIENTO")
# le = preprocessing.LabelEncoder()

# # Identificar columnas categóricas que necesitan transformación (por ejemplo, columnas 1, 2, 3)
# columnas_categoricas = [2, 3, 4]

# for col in columnas_categoricas:
#     X[:, col] = le.fit_transform(X[:, col].astype(str))
#     #X_train[:, col] = le.fit_transform(X_train[:, col].astype(str))

# # Asegúrate de que las demás columnas de X_test y X_train sean numéricas
# # Si hay columnas que siguen teniendo valores no numéricos, tendrás que aplicar la transformación también

# # Convertir las demás columnas a float
# for col in range(X.shape[1]):
#     if col not in columnas_categoricas:
#         X[:, col] = X[:, col].astype(float)
#       #  X_train[:, col] = X_train[:, col].astype(float)

# y = le.fit_transform(y)
# #y_train = le.fit_transform(y_train)

# # Eliminamos NAN y convertimos los valores a float
# X = np.nan_to_num(X.astype(float))
# #X_train = np.nan_to_num(X_train.astype(float))
# y = np.nan_to_num(y.astype(float))
# #y_train = np.nan_to_num(y_train.astype(float))



# # División en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#LECTURA DE ARGUMENTOS
num_caract = 20
selector = 'RFE'
print("ARGUMENTOS: num_caract=" + str(num_caract) + " selector=" + str(selector))




#APERTURA FICHERO
print ("LECTURA DE FICHERO")
X_train = np.loadtxt("./datos/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/X_train.csv", delimiter=',')
y_train = np.loadtxt("./datos/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/y_train.csv", delimiter=',')
X_test = np.loadtxt("./datos/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/X_test.csv", delimiter=',')
y_test = np.loadtxt("./datos/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/y_test.csv", delimiter=',')



# Definir el modelo
model = RandomForestClassifier()

# Definir los hiperparámetros y sus posibles valores
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Ajustar el modelo
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo y los mejores hiperparámetros
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Mejores hiperparámetros:",best_params)