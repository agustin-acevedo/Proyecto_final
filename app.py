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
import subprocess
from flask import Flask, render_template, send_from_directory


app = Flask(__name__)

# Ruta principal para la página de inicio
@app.route('/')
def home():
    print("INICIO", flush=True)
    return render_template('index.html')

UPLOAD_FOLDER = 'uploads'  # Carpeta donde voy a guardar los csv subidos 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Crea la carpeta si no existe

@app.route('/upload-datasets', methods=['POST'])
def upload_datasets():
    try:
        train_file = request.files.get('trainFile')
        test_file = request.files.get('testFile')

        if not train_file or not test_file:
            return jsonify({'error': 'Faltan uno o ambos archivos'}), 400

        train_path = os.path.join(UPLOAD_FOLDER, 'train.csv')
        test_path = os.path.join(UPLOAD_FOLDER, 'test.csv')

        train_file.save(train_path)
        test_file.save(test_path)

        return jsonify({'success': True,'filePath': train_path, 'testPath': test_path}), 200
    except Exception as e:
        return jsonify({'filePath': train_file, 'testFile': test_file}), 500

#DECLARACIÓN CONSTANTES
NUM_CARACT = [5]

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    data = request.get_json()
    file_path = data.get('filePath') #Obtiene la ruta del conjunto de datos de entrenamiento 
    file_path_test = data.get('rutaTest') #Obtiene la ruta del conjunto de datos de prueba
    if not file_path or not  file_path_test:
        return jsonify({'error': 'Archivo no encontrado'}), 400
    
    try:
        print ("LECTURA DE FICHEROS",flush=True)
        df_train = pd.read_csv(file_path, header=None)  #Lee la ruta del conjunto de datos de entrenamiento 
       
        df_test = pd.read_csv(file_path_test, header=None) #Lee la ruta del conjunto de datos de prueba
        
        # Eliminar filas y columnas irrelevantes para el conjunto de datos de entrenamiento
        print(df_train.columns)
        duplicados_train = df_train.duplicated()
        print("Cantidad de filas duplicadas en el conjunto de datos de entrenamiento:", duplicados_train.sum())

        # # Eliminar columnas irrelevantes
        df_train = df_train.drop(df_train.columns[0], axis=1) # Columna que hace referencia al ID
        
        # Eliminar filas y columnas irrelevantes para el conjunto de datos de test
        print(df_test.columns)
        duplicados_test = df_test.duplicated()
        print("Cantidad de filas duplicadas en el conjunto de datos de entrenamiento:", duplicados_test.sum())

        # # Eliminar columnas irrelevantes
        df_test = df_test.drop(df_test.columns[0], axis=1) # Columna que hace referencia al ID
        
        np_data_training = np.asarray(df_train, dtype=None)
        np_data_test = np.asarray(df_test, dtype=None)
        
        X_train = np_data_training[:, 0:-2]  # Seleccionar todas las columnas menos las dos últimas en el conjunto de datos de entrenamiento
        y_train = np_data_training[:, -2]    # Seleccionar la penúltima columna (etiqueta como cadena) en el conjunto de datos de entrenamiento
        X_test = np_data_test[:, 0:-2]  # Seleccionar todas las columnas menos las dos últimas en el conjunto de datos de test
        y_test = np_data_test[:, -2]    # Seleccionar la penúltima columna (etiqueta como cadena) en el conjunto de datos de test
        print("INICIO DEL PREPROCESAMIENTO", flush=True)

    
        # Creamos el modelo a utilizar para eliminar las columnas categoricas
        le = preprocessing.LabelEncoder()

        # Identificar columnas categóricas que necesitan transformación (por ejemplo, columnas 1, 2, 3 que son datos no numericos)
        columnas_categoricas = [1, 2, 3] 

        for col in columnas_categoricas:
            X_train[:, col] = le.fit_transform(X_train[:, col].astype(str))
        
        for col in columnas_categoricas:
            X_test[:, col] = le.fit_transform(X_test[:, col].astype(str))


        # Convertir las demás columnas a float
        for col in range(X_train.shape[1]):
            if col not in columnas_categoricas:
                X_train[:, col] = X_train[:, col].astype(float)
              
        for col in range(X_test.shape[1]):
            if col not in columnas_categoricas:
                X_test[:, col] = X_test[:, col].astype(float)
                

        y_test = le.fit_transform(y_test)
        y_train = le.fit_transform(y_train)

        # Eliminamos NAN y convertimos los valores a float
        X_train = np.nan_to_num(X_train.astype(float))
        X_test = np.nan_to_num(X_test.astype(float))
        y_train = np.nan_to_num(y_train.astype(float))
        y_test = np.nan_to_num(y_test.astype(float))




        #DIVISIÓN DEL DATASET
       # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
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
            selector1 = RFE(estimador, n_features_to_select= NUM_CARACT[0], step=1)
          
            print ("SELECCION DE CARACTERÍSTICAS: " + str(num) + " CARACTERÍSTICAS, " + "SELECTOR RFE")
            selector1 = selector1.fit(X_train, y_train)
            print (selector1.ranking_)

            X_train1 = selector1.transform(X_train)
            X_test1 = selector1.transform(X_test)

            # CREAR DIRECTORIOS PARA GUARDAR DATOS PROCESADOS
            if not os.path.exists("./datos/caracteristicas" + str(num) + "selectorRFE"):
                os.mkdir("./datos/caracteristicas" + str(num) + "selectorRFE")
        

            #CREAR DIRECTORIOS PARA GUARDAR RESULTADOS
            if not os.path.exists("./resultados"):
                os.mkdir("./resultados")
            if not os.path.exists("./resultados/caracteristicas" + str(num) + "selectorRFE"):
                os.mkdir("./resultados/caracteristicas" + str(num) + "selectorRFE")
           
            #CREAR DIRECTORIOS PARA GUARDAR GRÁFICAS
            if not os.path.exists("./graficos"):
                os.mkdir("./graficos")
            if not os.path.exists("./graficos/caracteristicas" + str(num) + "selectorRFE"):
                os.mkdir("./graficos/caracteristicas" + str(num) + "selectorRFE")
          
            #GUARDAR EN DISCO
            np.savetxt("./datos/caracteristicas" + str(num) + "selectorRFE/X_train.csv", X_train1, delimiter=',')
            np.savetxt("./datos/caracteristicas" + str(num) + "selectorRFE/y_train.csv", y_train, delimiter=',')
            np.savetxt("./datos/caracteristicas" + str(num) + "selectorRFE/X_test.csv", X_test1, delimiter=',')
            np.savetxt("./datos/caracteristicas" + str(num) + "selectorRFE/y_test.csv", y_test, delimiter=',')
            with open("./datos/caracteristicas" + str(num) + "selectorRFE/ranking.npy",'wb') as f:
                np.save(f, selector1.ranking_)


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
        print("TESTEAR EL CLASIFICADOR")
        if os.path.exists('./resultados/descr_general.dat'):
            os.remove('./resultados/descr_general.dat')
        for num in NUM_CARACT:
            if os.path.exists("./resultados/caracteristicas" + str(num) + "selectorRFE/bacc_caracteristicasSelector.dat"):
                os.remove("./resultados/caracteristicas" + str(num) + "selectorRFE/bacc_caracteristicasSelector.dat")

        for num in NUM_CARACT:
            script_descriptor = open("./clasificadores/" + model_name + ".py")
            script = script_descriptor.read()
            sys.argv = [str(model_name) + ".py", int(num), 'RFE']
            exec(script)

        # Enviar el resultado al frontend
        return jsonify({
            'status': 'success'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/results', methods=['POST'])
def generar_graficos():
   # print()
    #data = request.json
    #modelo = data.get('modeloSeleccionado')
    try:
        # Ejecutar el script que genera los gráficos
        subprocess.run(['python', 'GenerarGraficos.py'], check=True)
        #script_descriptor = open("GenerarGraficos.py")
        #script = script_descriptor.read()
       # sys.argv = [str(modelo), 5, 'RFE']
       # exec(script)
        
        return jsonify({"mensaje": "Gráficos generados correctamente"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Error al generar gráficos: {str(e)}"}), 500


IMAGENES_DIR = "./graficos/caracteristicas"+str(NUM_CARACT[0])+"selectorRFE"
CLASIFICADORES = np.array(['DT','RF','Bagging']) 

titulos_imagenes = {
    "figBacc_t.png": "Comparación de Excantitud Balanceada de los modelos probados",
    "figP_t.png": "Comparación de la precisión de los modelos probados",
    "figTRT_t.png": "Comparación del tiempo de entrenamiento de los modelos probados",
    "figTST_t.png": "Comparación del tiempo de entrenamiento de los modelos probados",
    "tab"+CLASIFICADORES[0]+"-Metricas.png": "Metricas por clases para clasificador: Desicion Tree",
    "tab"+CLASIFICADORES[1]+"-Metricas.png": "Metricas por clases para clasificador: Random Forest",
    "tab"+CLASIFICADORES[2]+"-Metricas.png": "Metricas por clases para clasificador: Bagging"
}




@app.route('/resultados')
def resultados():
    imagenes = os.listdir(IMAGENES_DIR)
    imagenes_info = [{"nombre": img, "titulo": titulos_imagenes.get(img, "Título no disponible")} for img in imagenes]
    return render_template('reportes.html', imagenes=imagenes_info)


@app.route('/imagenes/<path:filename>')
def imagenes(filename):
    return send_from_directory(IMAGENES_DIR, filename)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


