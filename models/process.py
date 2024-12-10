
import os.path
import sys
import numpy as np
import csv
import time
from sklearn import tree
from sklearn import svm
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


import pickle

import codecs



#DECLARACIÓN CONSTANTES
NUM_CARACT = [20]
#CLASIFICADORES = ['DT','NB','ANN','SVM','RF','GBM','VC']
CLASIFICADORES = ['EnsembleLearning','Bagging','XGBoost']
#CLASIFICADORES = ['DT','NB']



#reader = csv.reader(open("./datos/UNSW_NB15_testing-set.csv"), delimiter=",")
#raw_data = list(reader)

#np_data = np.asarray(raw_data, dtype=None)

#X_us = np_data[:, 0:-2]  # Seleccionar todas las columnas menos las dos últimas
#y = np_data[:, -2]   # Seleccionar la penúltima columna (etiqueta como cadena)


def leerArchivo(file_path):
    print ("LECTURA DE FICHERO")
    with codecs.open("./uploads/UNSW_NB15_testing-set.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=",")
    
    try: 
        return reader
    except Exception as e:
        raise ValueError (f"Error al leer el archivo: {e}")




# print("LECTURA DE FICHERO DE TRAINING")

# # # Abrimos el archivo de training de la misma manera
# # with codecs.open("./datos/UNSW_NB15_training-set.csv", "r", encoding="utf-8-sig") as f:
# #     reader_training = csv.reader(f, delimiter=",")
# #     raw_data_training = list(reader_training)

# # np_data_training = np.asarray(raw_data_training, dtype=None)

# # X_train = np_data_training[:, 0:-2]  # Seleccionar todas las columnas menos las dos últimas
# # y_train = np_data_training[:, -2]    # Seleccionar la penúltima columna (etiqueta como cadena)





# # #AGRUPAMOS ETIQUETAS
# # print("AGRUPAMOS ETIQUETAS")
# # for i in range(0,len(y)):
# #     if (y[i] in ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']):
# #         y[i] = 'Probe'
# #     elif (y[i] in ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named']):
# #         y[i] = 'R2L'
# #     elif (y[i] in ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']):
# #         y[i] = 'U2R'
# #     elif (y[i] in ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'worm', 'mailbomb']):
# #         y[i] = 'DoS'
# #     elif (y[i] == 'normal'):
# #         y[i] = 'Normal'
# #     else:
# #         y[i] = 'Unknown'


def preprocesamiento(X, y):
    print("INICIO DEL PREPROCESAMIENTO")

    # Asumimos que las columnas categóricas están en las posiciones 1, 2 y 3, pero pueden variar según tu dataset
    # Si hay más columnas categóricas, puedes añadirlas al mismo proceso
    le = preprocessing.LabelEncoder()

    # Identificar columnas categóricas que necesitan transformación (por ejemplo, columnas 1, 2, 3)
    columnas_categoricas = [2, 3, 4]

    for col in columnas_categoricas:
        X[:, col] = le.fit_transform(X[:, col].astype(str))
        # X_train[:, col] = le.fit_transform(X_train[:, col].astype(str))

    # Asegúrate de que las demás columnas de X_test y X_train sean numéricas
    # Si hay columnas que siguen teniendo valores no numéricos, tendrás que aplicar la transformación también

    # Convertir las demás columnas a float
    for col in range(X.shape[1]):
        if col not in columnas_categoricas:
            X[:, col] = X[:, col].astype(float)
            # X_train[:, col] = X_train[:, col].astype(float)

    y = le.fit_transform(y)
    # y_train = le.fit_transform(y_train)

    # Eliminamos NAN y convertimos los valores a float
    X = np.nan_to_num(X.astype(float))
    # X_train = np.nan_to_num(X_train.astype(float))
    y = np.nan_to_num(y.astype(float))
    # y_train = np.nan_to_num(y_train.astype(float))
    
    return X, y



# # PREPROCESAMIENTO
# print("PREPROCESAMIENTO")

# # Asumimos que las columnas categóricas están en las posiciones 1, 2 y 3, pero pueden variar según tu dataset
# # Si hay más columnas categóricas, puedes añadirlas al mismo proceso
# le = preprocessing.LabelEncoder()

# # Identificar columnas categóricas que necesitan transformación (por ejemplo, columnas 1, 2, 3)
# columnas_categoricas = [2, 3, 4]

# for col in columnas_categoricas:
#     X[:, col] = le.fit_transform(X[:, col].astype(str))
#     # X_train[:, col] = le.fit_transform(X_train[:, col].astype(str))

# # Asegúrate de que las demás columnas de X_test y X_train sean numéricas
# # Si hay columnas que siguen teniendo valores no numéricos, tendrás que aplicar la transformación también

# # Convertir las demás columnas a float
# for col in range(X.shape[1]):
#     if col not in columnas_categoricas:
#         X[:, col] = X[:, col].astype(float)
#         # X_train[:, col] = X_train[:, col].astype(float)

# y = le.fit_transform(y)
# # y_train = le.fit_transform(y_train)

# # Eliminamos NAN y convertimos los valores a float
# X = np.nan_to_num(X.astype(float))
# # X_train = np.nan_to_num(X_train.astype(float))
# y = np.nan_to_num(y.astype(float))
# # y_train = np.nan_to_num(y_train.astype(float))




# #DIVISIÓN DEL DATASET
# print ("DIVISIÓN DEL DATASET")
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# print ("X_train, y_train:", X_train.shape, y_train.shape)
# print ("X_test, y_test:", X_test.shape, y_test.shape)

# # RESUMEN DE LOS DATOS Y GUARDADO EN DISCO
# unique_elements, counts_elements = np.unique(le.inverse_transform(y_test.astype(int)), return_counts=True)
# print ("Número de elementos de cada clase en el Train Set:")
# print(np.asarray((unique_elements, counts_elements)))
# with open("./datos/train_descr.txt","w") as f:
#     f.write(str(np.asarray((unique_elements, counts_elements))))

# unique_elements, counts_elements = np.unique(le.inverse_transform(y_test.astype(int)), return_counts=True)
# print ("Número de elementos de cada clase en el Test Set:")
# print(np.asarray((unique_elements, counts_elements)))
# with open("./datos/test_descr.txt","w") as f:
#     f.write(str(np.asarray((unique_elements, counts_elements))))





# for num in NUM_CARACT:
#     # SELECCION DE CARACTERÍSTICAS
#     print ("SELECCION DE CARACTERÍSTICAS: " + str(num) + " CARACTERÍSTICAS")
#     estimador = tree.DecisionTreeClassifier()
#     selector1 = RFE(estimador, n_features_to_select=int(num), step=1)
#     selector2 = PCA(n_components=int(num))

#     print ("SELECCION DE CARACTERÍSTICAS: " + str(num) + " CARACTERÍSTICAS, " + "SELECTOR RFE")
#     selector1 = selector1.fit(X_test, y_test)
#     print (selector1.ranking_)

#     print ("SELECCION DE CARACTERÍSTICAS: " + str(num) + " CARACTERÍSTICAS, " + "SELECTOR PCA")
#     selector2 = selector2.fit(X_train, y_train)


#     X_train1 = selector1.transform(X_train)
#     X_test1 = selector1.transform(X_test)
#     X_train2 = selector2.transform(X_train)
#     X_test2 = selector2.transform(X_test)

#     # CREAR DIRECTORIOS PARA GUARDAR DATOS PROCESADOS
#     if not os.path.exists("./datos/features" + str(num) + "selectorRFE"):
#         os.mkdir("./datos/features" + str(num) + "selectorRFE")
#     # if not os.path.exists("./datos/features" + str(num) + "selectorPCA"):
#     #     os.mkdir("./datos/features" + str(num) + "selectorPCA")

#     #CREAR DIRECTORIOS PARA GUARDAR RESULTADOS
#     if not os.path.exists("./resultados"):
#         os.mkdir("./resultados")
#     if not os.path.exists("./resultados/features" + str(num) + "selectorRFE"):
#         os.mkdir("./resultados/features" + str(num) + "selectorRFE")
#     # if not os.path.exists("./resultados/features" + str(num) + "selectorPCA"):
#     #     os.mkdir("./resultados/features" + str(num) + "selectorPCA")

#     #CREAR DIRECTORIOS PARA GUARDAR GRÁFICAS
#     if not os.path.exists("./graficos"):
#         os.mkdir("./graficos")
#     if not os.path.exists("./graficos/features" + str(num) + "selectorRFE"):
#         os.mkdir("./graficos/features" + str(num) + "selectorRFE")
#     # if not os.path.exists("./graficos/features" + str(num) + "selectorPCA"):
#     #     os.mkdir("./graficos/features" + str(num) + "selectorPCA")


#     #GUARDAR EN DISCO
#     np.savetxt("./datos/features" + str(num) + "selectorRFE/X_train.csv", X_train1, delimiter=',')
#     np.savetxt("./datos/features" + str(num) + "selectorRFE/y_train.csv", y_train, delimiter=',')
#     np.savetxt("./datos/features" + str(num) + "selectorRFE/X_test.csv", X_test1, delimiter=',')
#     np.savetxt("./datos/features" + str(num) + "selectorRFE/y_test.csv", y_test, delimiter=',')
#     with open("./datos/features" + str(num) + "selectorRFE/ranking.npy",'wb') as f:
#         np.save(f, selector1.ranking_)


#     # np.savetxt("./datos/features" + str(num) + "selectorPCA/X_train.csv", X_train2, delimiter=',')
#     # np.savetxt("./datos/features" + str(num) + "selectorPCA/y_train.csv", y_train, delimiter=',')
#     # np.savetxt("./datos/features" + str(num) + "selectorPCA/X_test.csv", X_test2, delimiter=',')
#     # np.savetxt("./datos/features" + str(num) + "selectorPCA/y_test.csv", y_test, delimiter=',')


# #CREAR DIRECTORIO PARA GUARDAR MODELOS
# if not os.path.exists("./modelos"):
#     os.mkdir("./modelos")
# pickle.dump(le, open('./modelos/le.sav', 'wb'))





# #TESTEAR LOS CLASIFICADORES
# print("TESTEAR LOS CLASIFICADORES")
# if os.path.exists('./resultados/descr_general.dat'):
#   os.remove('./resultados/descr_general.dat')
# for num in NUM_CARACT:
#     # if os.path.exists("./resultados/features" + str(num) + "selectorPCA/bacc_FeaturesSelector.dat"):
#     #   os.remove("./resultados/features" + str(num) + "selectorPCA/bacc_FeaturesSelector.dat")
#     if os.path.exists("./resultados/features" + str(num) + "selectorRFE/bacc_FeaturesSelector.dat"):
#       os.remove("./resultados/features" + str(num) + "selectorRFE/bacc_FeaturesSelector.dat")

# for clf in CLASIFICADORES:
#     for num in NUM_CARACT:
#         script_descriptor = open("./clasificadores/" + clf + ".py")
#         script = script_descriptor.read()
#         sys.argv = [str(clf) + ".py", int(num), 'RFE']
#         exec(script)
#         # sys.argv = [str(clf) + ".py", int(num), 'PCA']
#         # exec(script)





# #EXTRAER GRÁFICAS
# print("EXTRAER GRÁFICAS")
# for num in NUM_CARACT:
#     script_descriptor = open("./ProcesarResultados.py", encoding="utf8")
#     script = script_descriptor.read()
#     sys.argv = ["ProcesarResultados.py", str(num), 'RFE']
#     exec(script)
# for num in NUM_CARACT:
#     script_descriptor = open("./ProcesarResultados.py", encoding="utf8")
#     script = script_descriptor.read()
#     sys.argv = ["ProcesarResultados.py", str(num), 'PCA']
#     exec(script)