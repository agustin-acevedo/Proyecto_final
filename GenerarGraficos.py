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
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt  # doctest: +SKIP
import pickle
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import seaborn as sns
from operator import itemgetter





#INICIO DEL PROGRAMA

CLASIFICADORES = np.array(['DT','RF','Bagging'])
METRICAS = np.array(['Fscore','Precision','Sensibilidad','Total'])
NUM_CARACT = [5]

#num_caract = sys.argv[1]
#selector = sys.argv[2]

#LECTURA DE ARGUMENTOS
num_caract = 5
selector = 'RFE'
print("ARGUMENTOS: num_caract=" + str(num_caract) + " selector=" + str(selector))




#APERTURA FICHERO
print ("LECTURA DE FICHEROS")
le = pickle.load(open('./modelos/le.sav', 'rb'))

CLASES = le.classes_




#Crear array con 'balanced_accuracy_score' de cada clasificador

baccs_t = np.array([])
for clf in CLASIFICADORES:
    file_path = "./resultados/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-bacc.dat"
    if not os.path.exists(file_path):
        print("El valor está vacío")
    else:
        with open(file_path,"r") as f:
            num = np.array([float(f.readline())])
            baccs_t = np.append(baccs_t, num)

#Crear array con 'precision' de cada clasificador

p_t = np.array([])
for clf in CLASIFICADORES:
    file_path = "./resultados/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-precision.dat"
    if not os.path.exists(file_path):
        print("El valor está vacío")
    else:
        with open(file_path,"r") as f:
            num = np.array([float(f.readline())])
            p_t = np.append(p_t, num)


#Crear array con 'Tiempo_entrenamiento' de cada clasificador

t_entrenamiento_t = np.array([])
for clf in CLASIFICADORES:
    file_path = "./resultados/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-Tiempo_entrenamiento.dat"
    if not os.path.exists(file_path):
        print("El valor está vacío")
    else:
        with open(file_path,"r") as f:
            num = np.array([float(f.readline())])
            t_entrenamiento_t = np.append(t_entrenamiento_t, num)


#Crear array con 'Tiempo_clasificacion' de cada clasificador

t_clasificacion_t = np.array([])
for clf in CLASIFICADORES:
    file_path = "./resultados/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-Tiempo_clasificacion.dat"
    if not os.path.exists(file_path):
        print("El valor está vacío")
    else:
        with open(file_path,"r") as f:
            num = np.array([float(f.readline())])
            t_clasificacion_t = np.append(t_clasificacion_t, num)



def guardar_grafico(x, y, titulo, nombre_archivo, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    color = ['lightblue','purple', 'red']
    plt.bar(x, y, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
                
    plt.savefig(nombre_archivo, dpi=300)
    plt.close()
    print(f"Guardado: {nombre_archivo}")
#CREACIÓN DE TABLAS
print ("CREACIÓN DE TABLAS")
#Crear tabla por cada clasificador mostrando las diferentes métricas
contador = 0
for clf in CLASIFICADORES:
    contador =+ 1
    output_dir = f"./graficos/caracteristicas{num_caract}selector{selector}/"
    matriz_metricas = np.empty(shape=(len(METRICAS), len(le.classes_) ))
    
    for i in range(0,len(METRICAS)):
        file_path = "./resultados/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-" + METRICAS[i] + ".npy"
        if not os.path.exists(file_path):
            print("El valor está vacío")
        else:
            v_metricas = np.load(file_path)
            matriz_metricas[i] = v_metricas
             #METODO QUE FUNICONA 
            # Convertir los datos a DataFrame
            df = pd.DataFrame(matriz_metricas, index=METRICAS, columns=le.classes_)
            df = df.T
            
            #OTRO METODO 
            sns.set_theme(style="whitegrid", palette="pastel")  # Estilo más limpio
            
            fig, ax = plt.subplots(figsize=(len(df.columns)/2, len(df.index)/2))
            ax.axis('tight')
            ax.axis('off')

            tabla = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, 
                            cellLoc='center', loc='center', colWidths=[0.2]*len(df.columns))

            tabla.auto_set_font_size(False)
            
            tabla.set_fontsize(9)
            tabla.auto_set_column_width([i for i in range(len(df.columns))])            
            #guardar_grafico(clf, baccs_t, "Exactitud Balanceada", output_dir + "figBacc_t.png", "Clasificadores", "Exactitud balanceada")
            #guardar_grafico(clf, p_t, "Precisión", output_dir + "figP_t.png", "Clasificadores", "Precisión")
            #guardar_grafico(clf, t_entrenamiento_t / 60, "Tiempo de Entrenamiento", output_dir + "figTRT_t.png", "Clasificadores", "Tiempo de entrenamiento (min)")
            #guardar_grafico(clf, t_clasificacion_t, "Tiempo de Clasificación", output_dir + "figTST_t.png", "Clasificadores", "Tiempo de clasificación (seg)")
            plt.savefig("./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) +"/"+ f"tab{clf}-Metricas.png", dpi=300, bbox_inches='tight')
            plt.close()

          
# #CREACIÓN DE GRÁFICAS#     
print ("CREACIÓN DE GRÁFICAS")

    
#Crear gráficos con Matplotlib
output_dir = f"./graficos/caracteristicas{num_caract}selector{selector}/"
directorio_img_DT = "./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) +"/"+ f"tabDT-Metricas.png"
directorio_img_RF = "./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) +"/"+ f"tabRF-Metricas.png"
directorio_img_BG = "./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) +"/"+ f"tabBagging-Metricas.png"

if os.path.exists(directorio_img_DT) and os.path.exists(directorio_img_RF) and os.path.exists(directorio_img_BG): 
    guardar_grafico(CLASIFICADORES, baccs_t, "Exactitud Balanceada", output_dir + "figBacc_t.png", "Clasificadores", "Exactitud balanceada")
    guardar_grafico(CLASIFICADORES, p_t, "Precisión", output_dir + "figP_t.png", "Clasificadores", "Precisión")
    guardar_grafico(CLASIFICADORES, t_entrenamiento_t / 60, "Tiempo de Entrenamiento", output_dir + "figTRT_t.png", "Clasificadores", "Tiempo de entrenamiento (min)")
    guardar_grafico(CLASIFICADORES, t_clasificacion_t, "Tiempo de Clasificación", output_dir + "figTST_t.png", "Clasificadores", "Tiempo de clasificación (seg)")
else:
    if os.path.exists(directorio_img_DT) and not os.path.exists(directorio_img_RF) and not os.path.exists(directorio_img_BG): 
        guardar_grafico(CLASIFICADORES[0], baccs_t, "Exactitud Balanceada", output_dir + "figBacc_t.png", "Clasificadores", "Exactitud balanceada")
        guardar_grafico(CLASIFICADORES[0], p_t, "Precisión", output_dir + "figP_t.png", "Clasificadores", "Precisión")
        guardar_grafico(CLASIFICADORES[0], t_entrenamiento_t / 60, "Tiempo de Entrenamiento", output_dir + "figTRT_t.png", "Clasificadores", "Tiempo de entrenamiento (min)")
        guardar_grafico(CLASIFICADORES[0], t_clasificacion_t, "Tiempo de Clasificación", output_dir + "figTST_t.png", "Clasificadores", "Tiempo de clasificación (seg)")
    else:
        if not os.path.exists(directorio_img_DT) and os.path.exists(directorio_img_RF) and not os.path.exists(directorio_img_BG): 
            guardar_grafico(CLASIFICADORES[1], baccs_t, "Exactitud Balanceada", output_dir + "figBacc_t.png", "Clasificadores", "Exactitud balanceada")
            guardar_grafico(CLASIFICADORES[1], p_t, "Precisión", output_dir + "figP_t.png", "Clasificadores", "Precisión")
            guardar_grafico(CLASIFICADORES[1], t_entrenamiento_t / 60, "Tiempo de Entrenamiento", output_dir + "figTRT_t.png", "Clasificadores", "Tiempo de entrenamiento (min)")
            guardar_grafico(CLASIFICADORES[1], t_clasificacion_t, "Tiempo de Clasificación", output_dir + "figTST_t.png", "Clasificadores", "Tiempo de clasificación (seg)")
        else:
            if not os.path.exists(directorio_img_DT) and not os.path.exists(directorio_img_RF) and os.path.exists(directorio_img_BG): 
                guardar_grafico(CLASIFICADORES[2], baccs_t, "Exactitud Balanceada", output_dir + "figBacc_t.png", "Clasificadores", "Exactitud balanceada")
                guardar_grafico(CLASIFICADORES[2], p_t, "Precisión", output_dir + "figP_t.png", "Clasificadores", "Precisión")
                guardar_grafico(CLASIFICADORES[2], t_entrenamiento_t / 60, "Tiempo de Entrenamiento", output_dir + "figTRT_t.png", "Clasificadores", "Tiempo de entrenamiento (min)")
                guardar_grafico(CLASIFICADORES[2], t_clasificacion_t, "Tiempo de Clasificación", output_dir + "figTST_t.png", "Clasificadores", "Tiempo de clasificación (seg)")
            else:
                if os.path.exists(directorio_img_DT) and not os.path.exists(directorio_img_RF) and  os.path.exists(directorio_img_BG):
                    getter = itemgetter(0, 2)
                    resultado = getter(CLASIFICADORES)
                    guardar_grafico(resultado, baccs_t, "Exactitud Balanceada", output_dir + "figBacc_t.png", "Clasificadores", "Exactitud balanceada")
                    guardar_grafico(resultado, p_t, "Precisión", output_dir + "figP_t.png", "Clasificadores", "Precisión")
                    guardar_grafico(resultado, t_entrenamiento_t / 60, "Tiempo de Entrenamiento", output_dir + "figTRT_t.png", "Clasificadores", "Tiempo de entrenamiento (min)")
                    guardar_grafico(resultado, t_clasificacion_t, "Tiempo de Clasificación", output_dir + "figTST_t.png", "Clasificadores", "Tiempo de clasificación (seg)")
                else:
                    if not os.path.exists(directorio_img_DT) and  os.path.exists(directorio_img_RF) and  os.path.exists(directorio_img_BG):
                        guardar_grafico(CLASIFICADORES[1:2], baccs_t, "Exactitud Balanceada", output_dir + "figBacc_t.png", "Clasificadores", "Exactitud balanceada")
                        guardar_grafico(CLASIFICADORES[1:2], p_t, "Precisión", output_dir + "figP_t.png", "Clasificadores", "Precisión")
                        guardar_grafico(CLASIFICADORES[1:2], t_entrenamiento_t / 60, "Tiempo de Entrenamiento", output_dir + "figTRT_t.png", "Clasificadores", "Tiempo de entrenamiento (min)")
                        guardar_grafico(CLASIFICADORES[1:2], t_clasificacion_t, "Tiempo de Clasificación", output_dir + "figTST_t.png", "Clasificadores", "Tiempo de clasificación (seg)")
            
        
