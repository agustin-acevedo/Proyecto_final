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





#INICIO DEL PROGRAMA

#CLASIFICADORES = np.array(['DT','NB','ANN','SVM','RF','GBM','VC'])
CLASIFICADORES = np.array(['DT', 'GBM', 'RF'])
METRICAS = np.array(['Fscore','Precision','Sensibilidad','Total'])
NUM_CARACT = [20]

#LECTURA DE ARGUMENTOS
num_caract = 20
selector = 'RFE'
print("ARGUMENTOS: num_caract=" + str(num_caract) + " selector=" + str(selector))




#APERTURA FICHERO
print ("LECTURA DE FICHEROS")
le = pickle.load(open('./modelos/le.sav', 'rb'))

CLASES = le.classes_


    #CREAR DIRECTORIOS PARA GUARDAR GRÁFICAS
if not os.path.exists("./graficos"):
    os.mkdir("./graficos")
if not os.path.exists("./graficos/caracteristicas" + str(num_caract) + "selectorRFE"):
    os.mkdir("./graficos/caracteristicas" + str(num_caract) + "selectorRFE")
    # if not os.path.exists("./graficos/caracteristicas" + str(num) + "selectorPCA"):
    #     os.mkdir("./graficos/caracteristicas" + str(num) + "selectorPCA")
# compararPCA = np.array([])
# for num in NUM_CARACT:
#     filepath = "./resultados/caracteristicas" + str(num) + "selectorPCA/bacc_FeaturesSelector.dat"
#     with open(filepath) as fp:
#        linea = fp.readline()
#        suma = float(linea)
#        cnt = 1
#        while linea:
#            linea = fp.readline()
#            if linea == '':
#                break
#            suma = suma + float(linea)
#            cnt = cnt + 1
#     media = suma/cnt
#     compararPCA = np.append(compararPCA, media)

# compararRFE = np.array([])
# for num in NUM_CARACT:
#     filepath = "./resultados/caracteristicas" + str(num) + "selectorRFE/bacc_FeaturesSelector.dat"
#     with open(filepath) as fp:
#        linea = fp.readline()
#        suma = float(linea)
#        cnt = 1
#        while linea:
#            linea = fp.readline()
#            if linea == '':
#                break
#            suma = suma + float(linea)
#            cnt = cnt + 1
#     media = suma/cnt
#     compararRFE = np.append(compararRFE, media)



#Crear array con 'balanced_accuracy_score' de cada clasificador

baccs_t = np.array([])
for clf in CLASIFICADORES:
    file_path = "./resultados/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-bacc.dat"
    with open(file_path,"r") as f:
        num = np.array([float(f.readline())])
        baccs_t = np.append(baccs_t, num)

#Crear array con 'precision' de cada clasificador

p_t = np.array([])
for clf in CLASIFICADORES:
    file_path = "./resultados/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-precision.dat"
    with open(file_path,"r") as f:
        num = np.array([float(f.readline())])
        p_t = np.append(p_t, num)


#Crear array con 'Tiempo_entrenamiento' de cada clasificador

t_entrenamiento_t = np.array([])
for clf in CLASIFICADORES:
    file_path = "./resultados/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-Tiempo_entrenamiento.dat"
    with open(file_path,"r") as f:
        num = np.array([float(f.readline())])
        t_entrenamiento_t = np.append(t_entrenamiento_t, num)


#Crear array con 'Tiempo_clasificacion' de cada clasificador

t_clasificacion_t = np.array([])
for clf in CLASIFICADORES:
    file_path = "./resultados/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-Tiempo_clasificacion.dat"
    with open(file_path,"r") as f:
        num = np.array([float(f.readline())])
        t_clasificacion_t = np.append(t_clasificacion_t, num)



print("PARA PROBAR SI SE ESTA EJECUTANDO ESTO: ",t_clasificacion_t)
print("PARA PROBAR SI SE ESTA EJECUTANDO ESTO: ",t_entrenamiento_t)
print("PARA PROBAR SI SE ESTA EJECUTANDO ESTO: ",p_t)
print("PARA PROBAR SI SE ESTA EJECUTANDO ESTO: ",baccs_t)
print("HASTA ACA FUNCIONA ")


#CREACIÓN DE TABLAS
print ("CREACIÓN DE TABLAS")
#Crear tabla por cada clasificador mostrando las diferentes métricas
for clf in CLASIFICADORES:
    matriz_metricas = np.empty(shape=(len(METRICAS), len(le.classes_) ))
    print("HASTA ACA FUNICONA 1")
    for i in range(0,len(METRICAS)):
        file_path = "./resultados/caracteristicas" + str(num_caract) + "selector" + str(selector) + "/" + clf + "-" + METRICAS[i] + ".npy"
        v_metricas = np.load(file_path)
        matriz_metricas[i] = v_metricas
        print("HASTA ACA FUNICONA 2")

    # fig = go.Figure(data=[go.Table(header=dict(values=np.append(['Clase-Metrica'],METRICAS)),
    #                  cells=dict(  values=np.transpose(np.concatenate((le.classes_[:,None],np.transpose(matriz_metricas)),axis=1))  ))
    #                      ])
    # fig.write_html("./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) + "/tab" + clf + "-PRUEBA.html")
    # print("Tabla guardada en HTML correctamente.")

    # #fig.write_image("./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) + "/tab" + clf + "-Metricas.png", engine="kaleido")
    
    #METODO QUE FUNICONA 
    # Convertir los datos a DataFrame
    df = pd.DataFrame(matriz_metricas, index=METRICAS, columns=le.classes_)
    df = df.T
    # # Crear figura
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.axis('tight')
    # ax.axis('off')

    # # Crear tabla
    # ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')

    # # Guardar imagen
    # plt.savefig("./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) + f"tab{clf}-Metricas.png")
    # plt.close()
    # print("Tabla guardada correctamente en Matplotlib.")
    
    # OTRO METODO QUE FUNCIONA 
    # plt.figure(figsize=(len(le.classes_)/2, len(METRICAS)/2))
    # sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar=False)
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=0)

    # plt.savefig("./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) + f"tab{clf}-Metricas.png", dpi=300, bbox_inches='tight')
    # plt.close()
    #--------------------------------------------------------------------------------------------------------------------------------------------------
    
    # plt.figure(figsize=(len(le.classes_)/2, len(METRICAS)/2))

    # # Crear tabla con Seaborn
    # ax = plt.gca()
    # ax.axis('tight')
    # ax.axis('off')

    # tabla = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, 
    #                 cellLoc='center', loc='center', colWidths=[0.2]*len(df.columns))

    # tabla.auto_set_font_size(False)
    # tabla.set_fontsize(8)  # Reducir fuente para que entre mejor
    # tabla.auto_set_column_width([i for i in range(len(df.columns))])

    # plt.savefig("./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) + f"tab{clf}-Metricas.png", dpi=300, bbox_inches='tight')
    # plt.close()
    
    sns.set_theme(style="whitegrid")  # Estilo más limpio

    fig, ax = plt.subplots(figsize=(len(df.columns)/2, len(df.index)/2))
    ax.axis('tight')
    ax.axis('off')

    tabla = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, 
                    cellLoc='center', loc='center', colWidths=[0.2]*len(df.columns))

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.auto_set_column_width([i for i in range(len(df.columns))])

    plt.savefig("./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) +"/"+ f"tab{clf}-Metricas.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("HASTA ACA FUNICONA 3")






#CREACIÓN DE GRÁFICAS
print ("CREACIÓN DE GRÁFICAS")

import matplotlib.pyplot as plt


def guardar_grafico(x, y, titulo, nombre_archivo, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    plt.bar(x, y, color='royalblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plt.savefig(nombre_archivo, dpi=300)
    plt.close()
    print(f"Guardado: {nombre_archivo}")

# Crear gráficos con Matplotlib
output_dir = f"./graficos/caracteristicas{num_caract}selector{selector}/"

guardar_grafico(CLASIFICADORES, baccs_t, "Exactitud Balanceada", output_dir + "figBacc_t.png", "Clasificadores", "Exactitud balanceada")
guardar_grafico(CLASIFICADORES, p_t, "Precisión", output_dir + "figP_t.png", "Clasificadores", "Precisión")
guardar_grafico(CLASIFICADORES, t_entrenamiento_t / 60, "Tiempo de Entrenamiento", output_dir + "figTRT_t.png", "Clasificadores", "Tiempo de entrenamiento (min)")
guardar_grafico(CLASIFICADORES, t_clasificacion_t /60, "Tiempo de Clasificación", output_dir + "figTST_t.png", "Clasificadores", "Tiempo de clasificación (min)")



# #GRAFICAS DE COLUMNAS
# figBacc_t = go.Figure([go.Bar(x=CLASIFICADORES, y=baccs_t)])
# figBacc_t.update_layout(
#     xaxis_title="Clasificadores",
#     yaxis_title="Exactitud balanceada"
# )

# figBacc_t.write_image("./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) + "/figBacc_t.png", engine="orca")


# figP_t = go.Figure([go.Bar(x=CLASIFICADORES, y=p_t)])
# figP_t.update_layout(
#     xaxis_title="Clasificadores",
#     yaxis_title="Precisión"
# )

# figP_t.write_image("./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) + "/figP_t.png", engine="orca")

# figTRT_t  = go.Figure([go.Bar(x=CLASIFICADORES, y=(t_entrenamiento_t/60))])
# figTRT_t.update_layout(
#     xaxis_title="Clasificadores",
#     yaxis_title="Tiempo de entrenamiento (min)"
# )

# figTRT_t.write_image("./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) + "/figTRT_t.png", engine="orca")

# figTST_t = go.Figure([go.Bar(x=CLASIFICADORES, y=t_clasificacion_t)])
# figTST_t.update_layout(
#     xaxis_title="Clasificadores",
#     yaxis_title="Tiempo de clasificación (s)"
# )

# figTST_t.write_image("./graficos/caracteristicas"  + str(num_caract) + "selector" + str(selector) + "/figTST_t.png", engine="orca")





# figCompararNumCaract = go.Figure([go.Scatter()])
# figCompararNumCaract.update_layout(
#     xaxis_title="Clasificadores",
#     yaxis_title="E"
# )
# figCompararNumCaract.add_trace(go.Scatter(x=NUM_CARACT, y=compararRFE,
#                     name='RFE'))
# figCompararNumCaract.add_trace(go.Scatter(x=NUM_CARACT, y=compararPCA,
#                     name='PCA',
#                     line=dict(color='firebrick', width=4,
#                               dash='dash')))
# figCompararNumCaract.write_image("./graficos/figCompararNumCaract.png")