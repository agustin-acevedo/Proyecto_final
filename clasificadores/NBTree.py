import os.path
import sys
import numpy as np
import pandas as pd
import csv
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import pickle


class NBTree:
    def __init__(self, max_depth=None):
        self.tree = DecisionTreeClassifier(max_depth=max_depth)
        self.leaf_models = {}

    def fit(self, X, y):
        self.tree.fit(X, y)
        self._fit_leaf_models(X, y)

    def _fit_leaf_models(self, X, y):
        leaf_indices = self.tree.apply(X)
        unique_leaf_indices = np.unique(leaf_indices)
        
        for leaf_index in unique_leaf_indices:
            leaf_mask = (leaf_indices == leaf_index)
            X_leaf = X[leaf_mask]
            y_leaf = y[leaf_mask]
            nb_model = GaussianNB()
            nb_model.fit(X_leaf, y_leaf)
            self.leaf_models[leaf_index] = nb_model

    def predict(self, X):
        leaf_indices = self.tree.apply(X)
        y_pred = np.zeros(X.shape[0], dtype=int)
        
        for leaf_index in np.unique(leaf_indices):
            leaf_mask = (leaf_indices == leaf_index)
            X_leaf = X[leaf_mask]
            nb_model = self.leaf_models[leaf_index]
            y_pred[leaf_mask] = nb_model.predict(X_leaf)
        
        return y_pred

#INICIO DEL PROGRAMA
print("EVALUACIÓN DEL CLASIFICADOR - " + sys.argv[0] )




#DECLARACIÓN CONSTANTES
L_CLF = 'NBTree'




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


#FASE DE ENTRENAMIENTO
print ("FASE DE ENTRENAMIENTO")
t_inicio_entrenamiento = time.time()
# Entrenar el NBTree
nb_tree = NBTree(max_depth=3)
nb_tree.fit(X_train, y_train)
t_fin_entrenamiento = time.time()

print ("Training time: " + str(t_fin_entrenamiento - t_inicio_entrenamiento))

#FASE DE CLASIFICACIÓN
print ("FASE DE CLASIFICACIÓN")
t_inicio_clasif = time.time()
y_pred = nb_tree.predict(X_test)

t_fin_clasif = time.time()

print ("Testing time: " + str(t_fin_clasif - t_inicio_clasif))

#OBTENCIÓN DE MÉTRICAS DE RENDIMIENTO
print ("OBTENCIÓN DE MÉTRICAS DE RENDIMIENTO")
p, recall, fscore, support = precision_recall_fscore_support(y_test,y_pred, average='micro')
bacc = balanced_accuracy_score(y_pred, y_test)

y_test = le.inverse_transform(y_test.astype(int))
y_pred = le.inverse_transform(y_pred.astype(int))
v_p, v_recall, v_fscore, v_support = precision_recall_fscore_support(y_test,y_pred, average=None, labels = le.classes_)

cnf_matrix = confusion_matrix(y_test,y_pred)

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
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/bacc_FeaturesSelector.dat',"a") as f:
    f.write(str(float(bacc)) + " \n")


#Confusion Matrix y métricas derivadas
with open('./resultados/features' + str(num_caract) + 'selector' + str(selector) + '/' + L_CLF + '-confusion_matrix.npy','wb') as f:
    np.save(f, confusion_matrix)