from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os.path
from models.process import leerArchivo, preprocesamiento
import numpy as np
import pickle
import codecs
import sys
import numpy as np
import csv
import time
from sklearn import tree
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


app = Flask(__name__)

# Ruta principal para la página de inicio
@app.route('/')
def home():
    #return render_template('index.html')
    return render_template('inicio.html')
# raw_data_test = list(reader)

#     np_data_test = np.asarray(raw_data_test, dtype=None)

#     X = np_data_test[:, 0:-2]  # Seleccionar todas las columnas menos las dos últimas
#     y = np_data_test[:, -2]    # Seleccionar la penúltima columna (etiqueta como cadena)


#@app.route('/upload', methods=['POST'])
#def upload_file():
#    if 'file' not in request.files:
#        return jsonify({'error': 'No file uploaded'}), 400
#
#    file = request.files['file']
#    if file:
#        file.save(f'uploads/{file.filename}')  # Guardar el archivo en la carpeta 'uploads'
#        return jsonify({'message': 'File uploaded successfully'}), 200
#    return jsonify({'error': 'Invalid file'}), 400

UPLOAD_FOLDER = 'uploads'  # Carpeta donde voy a guardar los csv subidos 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Crea la carpeta si no existe

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Guarda el archivo en el servidor
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Devuelve la ruta del archivo
    return jsonify({'filePath': file_path}), 200



#DECLARACIÓN CONSTANTES
NUM_CARACT = [20]

CLASIFICADORES = ['EnsembleLearning','Bagging','XGBoost']

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    data = request.get_json()
    file_path = data.get('filePath')

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'Archivo no encontrado'}), 400
    
    try:
        print ("LECTURA DE FICHERO")
        with codecs.open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=",")
            raw_data = list(reader)
        
        np_data_training = np.asarray(raw_data, dtype=None)

        X = np_data_training[:, 0:-2]  # Seleccionar todas las columnas menos las dos últimas
        y = np_data_training[:, -2]    # Seleccionar la penúltima columna (etiqueta como cadena)
        
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




        #DIVISIÓN DEL DATASET
        print ("El conjunto de datos se esta dividiendo..")
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        print ("X_train, y_train:", X_train.shape, y_train.shape)
        print ("X_test, y_test:", X_test.shape, y_test.shape)
        
        #Se guarda los datos en el disco
        print("Los datos se estan guardando en el disco...")
        
        unique_elements, counts_elements = np.unique(le.inverse_transform(y_train.astype(int)), return_counts=True)
        print ("Número de elementos de cada clase en el Train Set:")
        print(np.asarray((unique_elements, counts_elements)))
       
        # Crear la carpeta si no existe
        os.makedirs("datos", exist_ok=True)

        # Ruta completa del archivo
        ruta_archivo = os.path.join("datos", "train_descr.txt")

        # Escribir en el archivo
        with open(ruta_archivo, "w") as archivo:
            archivo.write(str(np.asarray((unique_elements, counts_elements))))

        unique_elements, counts_elements = np.unique(le.inverse_transform(y_test.astype(int)), return_counts=True)
        print ("Número de elementos de cada clase en el Test Set:")
        print(np.asarray((unique_elements, counts_elements)))
         # Crear la carpeta si no existe
        os.makedirs("datos", exist_ok=True)

        # Ruta completa del archivo
        ruta_archivo = os.path.join("datos", "test_descr.txt")

        # Escribir en el archivo
        with open(ruta_archivo, "w") as archivo:
            archivo.write(str(np.asarray((unique_elements, counts_elements))))
                
        for num in NUM_CARACT:
            # SELECCION DE CARACTERÍSTICAS
            print ("SELECCION DE CARACTERÍSTICAS: " + str(num) + " CARACTERÍSTICAS")
            estimador = tree.DecisionTreeClassifier()
            selector1 = RFE(estimador, n_features_to_select= 20, step=1)
            #selector2 = PCA(n_components=int(num))

            print ("SELECCION DE CARACTERÍSTICAS: " + str(num) + " CARACTERÍSTICAS, " + "SELECTOR RFE")
            selector1 = selector1.fit(X_test, y_test)
            print (selector1.ranking_)

            #print ("SELECCION DE CARACTERÍSTICAS: " + str(num) + " CARACTERÍSTICAS, " + "SELECTOR PCA")
            #selector2 = selector2.fit(X_train, y_train)


            X_train1 = selector1.transform(X_train)
            X_test1 = selector1.transform(X_test)
            #X_train2 = selector2.transform(X_train)
            #X_test2 = selector2.transform(X_test)

            # CREAR DIRECTORIOS PARA GUARDAR DATOS PROCESADOS
            if not os.path.exists("./datos/caracteristicas" + str(num) + "selectorRFE"):
                os.mkdir("./datos/caracteristicas" + str(num) + "selectorRFE")
            # if not os.path.exists("./datos/caracteristicas" + str(num) + "selectorPCA"):
            #     os.mkdir("./datos/caracteristicas" + str(num) + "selectorPCA")

            #CREAR DIRECTORIOS PARA GUARDAR RESULTADOS
            if not os.path.exists("./resultados"):
                os.mkdir("./resultados")
            if not os.path.exists("./resultados/caracteristicas" + str(num) + "selectorRFE"):
                os.mkdir("./resultados/caracteristicas" + str(num) + "selectorRFE")
            # if not os.path.exists("./resultados/caracteristicas" + str(num) + "selectorPCA"):
            #     os.mkdir("./resultados/caracteristicas" + str(num) + "selectorPCA")

            #CREAR DIRECTORIOS PARA GUARDAR GRÁFICAS
            if not os.path.exists("./graficos"):
                os.mkdir("./graficos")
            if not os.path.exists("./graficos/caracteristicas" + str(num) + "selectorRFE"):
                os.mkdir("./graficos/caracteristicas" + str(num) + "selectorRFE")
            # if not os.path.exists("./graficos/caracteristicas" + str(num) + "selectorPCA"):
            #     os.mkdir("./graficos/caracteristicas" + str(num) + "selectorPCA")


            #GUARDAR EN DISCO
            np.savetxt("./datos/caracteristicas" + str(num) + "selectorRFE/X_train.csv", X_train1, delimiter=',')
            np.savetxt("./datos/caracteristicas" + str(num) + "selectorRFE/y_train.csv", y_train, delimiter=',')
            np.savetxt("./datos/caracteristicas" + str(num) + "selectorRFE/X_test.csv", X_test1, delimiter=',')
            np.savetxt("./datos/caracteristicas" + str(num) + "selectorRFE/y_test.csv", y_test, delimiter=',')
            with open("./datos/caracteristicas" + str(num) + "selectorRFE/ranking.npy",'wb') as f:
                np.save(f, selector1.ranking_)


            # np.savetxt("./datos/caracteristicas" + str(num) + "selectorPCA/X_train.csv", X_train2, delimiter=',')
            # np.savetxt("./datos/caracteristicas" + str(num) + "selectorPCA/y_train.csv", y_train, delimiter=',')
            # np.savetxt("./datos/caracteristicas" + str(num) + "selectorPCA/X_test.csv", X_test2, delimiter=',')
            # np.savetxt("./datos/caracteristicas" + str(num) + "selectorPCA/y_test.csv", y_test, delimiter=',')


        #CREAR DIRECTORIO PARA GUARDAR MODELOS
        if not os.path.exists("./modelos"):
            os.mkdir("./modelos")
            pickle.dump(le, open('./modelos/le.sav', 'wb'))
        
        processed_data = {
            'message': 'Preprocesamiento completado',
            'status': 'success'
        }
        
        return jsonify(processed_data), 200
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")
        return jsonify({'message': 'Error durante el preprocesamiento', 'status': 'error'}), 500



# Ruta para entrenar el modelo
@app.route('/train', methods=['POST'])
def train():
     
    # Recibir el modelo seleccionado
    data = request.json
    model_name = data.get('model')

    if not model_name:
        return jsonify({'error': 'No se recibió el modelo seleccionado'}), 400
    
    try:
        #TESTEAR LOS CLASIFICADORES
        print("TESTEAR EL CLASIFICADOR")
        if os.path.exists('./resultados/descr_general.dat'):
            os.remove('./resultados/descr_general.dat')
        for num in NUM_CARACT:
            # if os.path.exists("./resultados/caracteristicas" + str(num) + "selectorPCA/bacc_caracteristicasSelector.dat"):
            #   os.remove("./resultados/caracteristicas" + str(num) + "selectorPCA/bacc_caracteristicasSelector.dat")
            if os.path.exists("./resultados/caracteristicas" + str(num) + "selectorRFE/bacc_caracteristicasSelector.dat"):
                os.remove("./resultados/caracteristicas" + str(num) + "selectorRFE/bacc_caracteristicasSelector.dat")

        for num in NUM_CARACT:
            script_descriptor = open("./clasificadores/" + model_name + ".py")
            script = script_descriptor.read()
            sys.argv = [str(model_name) + ".py", int(num), 'RFE']
            exec(script)
            # sys.argv = [str(model_name) + ".py", int(num), 'PCA']
            # exec(script)

        # Enviar el resultado al frontend
        return jsonify({
            'status': 'success'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
