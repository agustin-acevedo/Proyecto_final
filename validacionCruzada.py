import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing

# Cargar el dataset
df = pd.read_csv("./uploads/train.csv", header=None)
print(df.columns)
duplicados = df.duplicated()
print("Cantidad de filas duplicadas:", duplicados.sum())

# # Eliminar columnas irrelevantes
df = df.drop(df.columns[0], axis=1)


# Eliminar filas duplicadas si existieran
#df = df.drop_duplicates()


# Separaramos las variables predictoras y etiqueta
np_data_training = np.asarray(df, dtype=None)

X = np_data_training[:, 0:-2]  # Seleccionar todas las columnas menos las dos últimas
y = np_data_training[:, -2]    # Seleccionar la penúltima columna (etiqueta como cadena)



le = preprocessing.LabelEncoder() #Tecnica para eliminar las variables categoricas

# Identificar columnas categóricas que necesitan transformación (por ejemplo, columnas  1,2,3)
columnas_categoricas = [1, 2, 3]

for col in columnas_categoricas:
     X[:, col] = le.fit_transform(X[:, col].astype(str))
         


# Convertir las demás columnas a float
for col in range(X.shape[1]):
    if col not in columnas_categoricas:
        X[:, col] = X[:, col].astype(float)


y = le.fit_transform(y)


# Eliminamos NAN y convertimos los valores a float
X = np.nan_to_num(X.astype(float))  
y = np.nan_to_num(y.astype(float))

 
# Escalar características
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# Modelo base
model = DecisionTreeClassifier()

# Evaluar RFE con diferentes cantidades de características
feature_range = range(5, 41, 5)
mean_scores = []

for n in feature_range:
    rfe = RFE(estimator=model, n_features_to_select=n)
    X_rfe = rfe.fit_transform(X, y)
    
    # Validación cruzada con 5 folds
    scores = cross_val_score(model, X_rfe, y, cv=5, scoring='accuracy')
    mean_score = scores.mean()
    mean_scores.append(mean_score)
    print(f"Características: {n}, Accuracy Promedio: {mean_score:.4f}")

# Encontrar el número óptimo
best_n = feature_range[np.argmax(mean_scores)]
print(f"\nMejor cantidad de características: {best_n}")

# Graficar resultados
plt.figure(figsize=(10, 5))
plt.plot(feature_range, mean_scores, marker='o')
plt.title('Selección de características con RFE + Validación cruzada')
plt.xlabel('Número de características seleccionadas')
plt.ylabel('Accuracy promedio (CV)')
plt.grid(True)
plt.show()
