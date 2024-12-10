import os.path
import sys
import numpy as np
import pandas as pd
import csv
import time
from sklearn import tree
from sklearn import svm
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import pickle
from sklearn.utils import resample

#INICIO DEL PROGRAMA
print("EVALUACIÓN DEL CLASIFICADOR - " + sys.argv[0] )




#DECLARACIÓN CONSTANTES
L_CLF = 'DT2'




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
#PREPROCESAMIENTO
print("PREPROCESAMIENTO")

# Supongamos que 'X_train' es un numpy array y 'y_train' es un numpy array

# Convertir X_train a un DataFrame
X_train_df = pd.DataFrame(X_train)

#Convertir X_test en un Dataframe 
X_test_df = pd.DataFrame(X_test)

# Convertir y_train a una Serie de pandas
y_train_series = pd.Series(y_train, name='label')

# Convertir y_test a una serie de pandas 
y_test_series = pd.Series(y_test, name='label')


# Concatenar X_train e y_train para formar un DataFrame completo
data = pd.concat([X_train_df, y_train_series], axis=1)

# Concatenar X_test e y_test para formar un DataFrame completo
data_test = pd.concat([X_test_df, y_test_series], axis=1)

# Calcular la distribución de clases en el train set
class_distribution = y_train_series.value_counts()
print("Distribución de clases antes del balanceo en el train set:")
print(class_distribution)

# Calcular la distribución de clases en el train set
class_distribution_test = y_test_series.value_counts()
print("Distribución de clases antes del balanceo en el test set:")
print(class_distribution_test)

# Identificar la clase minoritaria
minority_class_label = class_distribution.idxmin()
print(f"La clase minoritaria en el train set es: {minority_class_label}")

# Identificar la clase minoritaria
minority_class_label_test = class_distribution_test.idxmin()
print(f"La clase minoritaria en el test set es: {minority_class_label_test}")

# Separar las clases mayoritarias y minoritarias
class_minority = data[data['label'] == minority_class_label]
class_majority = data[data['label'] != minority_class_label]

# Separar las clases mayoritarias y minoritarias en el test set 
class_minority_test = data_test[data_test['label'] == minority_class_label_test]
class_majority_test = data_test[data_test['label'] != minority_class_label_test]

# Sobremuestrear la clase minoritaria en el train set
class_minority_oversampled = resample(class_minority,
                                      replace=True,
                                      n_samples= 2000,
                                      random_state=123)

# Sobremuestrear la clase minoritaria en el test set
class_minority_oversampled_test = resample(class_minority_test,
                                      replace=True,
                                      n_samples= 2000,
                                      random_state=123)

# Combinar clases mayoritarias y minoritarias sobremuestreadas
data_balanced = pd.concat([class_majority, class_minority_oversampled])
# Combinar clases mayoritarias y minoritarias sobremuestreadas en el test set
data_balanced_test = pd.concat([class_majority_test, class_minority_oversampled_test])

# Separar características y etiquetas
X_train_balanced = data_balanced.drop('label', axis=1)
y_train_balanced = data_balanced['label']

# Separar características y etiquetas en el test set
X_test_balanced = data_balanced_test.drop('label', axis=1)
y_test_balanced = data_balanced_test['label']

# Verificar la nueva distribución de clases
print("Distribución de clases después del balanceo en el train set:")
print(y_train_balanced.value_counts())

# Verificar la nueva distribución de clases en el test set
print("Distribución de clases después del balanceo en el test set:")
print(y_test_balanced.value_counts())


#SELECCION DE CLASIFICADOR
clasificador = tree.DecisionTreeClassifier()




#PREPROCESAMIENTO
print("PREPROCESAMIENTO")

#Normalizamos
#X_train = preprocessing.normalize(X_train)
#X_test = preprocessing.normalize(X_test)

#Escalamos
#X_train = preprocessing.scale(X_train)
#X_test = preprocessing.scale(X_test)




#FASE DE ENTRENAMIENTO
print ("FASE DE ENTRENAMIENTO")
t_inicio_entrenamiento = time.time()
clasificador.fit(X_train_balanced, y_train_balanced)
t_fin_entrenamiento = time.time()

print ("Training time: " + str(t_fin_entrenamiento - t_inicio_entrenamiento))




#FASE DE CLASIFICACIÓN
print ("FASE DE CLASIFICACIÓN")
t_inicio_clasif = time.time()
y_pred = clasificador.predict(X_test_balanced)
t_fin_clasif = time.time()

print ("Testing time: " + str(t_fin_clasif - t_inicio_clasif))




#OBTENCIÓN DE MÉTRICAS DE RENDIMIENTO
print ("OBTENCIÓN DE MÉTRICAS DE RENDIMIENTO")
p, recall, fscore, support = precision_recall_fscore_support(y_test_balanced, y_pred, average='micro')
bacc = balanced_accuracy_score(y_pred, y_test_balanced)

y_test = le.inverse_transform(y_test_balanced.astype(int))
y_pred = le.inverse_transform(y_pred.astype(int))
v_p, v_recall, v_fscore, v_support = precision_recall_fscore_support(y_test, y_pred, average=None, labels = le.classes_)

cnf_matrix = confusion_matrix(y_test, y_pred)

#Imprimir resumen por pantalla
print ("Exactitud balanceada (Balanced accuracy score): ")
print (bacc)
print (classification_report(y_test,y_pred))




#GUARDAR RESULTADOS A DISCO
print ("GUARDAR RESULTADOS A DISCO")
if not os.path.exists("./resultados"):
    os.mkdir("./resultados")

#Métricas de rendimiento globales
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-bacc.dat',"w") as f:
    f.write(str(float(bacc)))
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-precision.dat',"w") as f:
    f.write(str(float(p)))
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-Tiempo_entrenamiento.dat',"w") as f:
    f.write(str(float(t_fin_entrenamiento - t_inicio_entrenamiento)))
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-Tiempo_clasificacion.dat',"w") as f:
    f.write(str(float(t_fin_clasif - t_inicio_clasif)))

#Metricas de rendimiento por clase
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-Precision.npy','wb') as f:
    np.save(f, v_p)
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-Sensibilidad.npy','wb') as f:
    np.save(f, v_recall)
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-Fscore.npy','wb') as f:
    np.save(f, v_fscore)
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-Total.npy','wb') as f:
    np.save(f, v_support)


#Métricas para la selección de características
with open('./resultados/descr_general.dat',"a") as f:
    f.write(L_CLF + " num_caract=" + str(num_caract) + " selector=" + str(selector) + " bacc=" + str(float(bacc)) + " p=" + str(float(p)) + " TRT=" + str(t_fin_entrenamiento - t_inicio_entrenamiento) + " \n")
with open('./resultados/numero_caracteristicas' + str(num_caract) + 'selector' + str(selector) + '/bacc_FeaturesSelector.dat',"a") as f:
    f.write(str(float(bacc)) + " \n")


#Confusion Matrix y métricas derivadas
with open('./resultados/numero_caracteristicas' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-confusion_matrix.npy','wb') as f:
    np.save(f, confusion_matrix)
