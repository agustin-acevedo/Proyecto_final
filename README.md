# Sistema de Comparación de Algoritmos de Machine Learning para Detección de Intrusiones

Este proyecto forma parte de mi tesis de Licenciatura en Sistemas. Su objetivo es **analizar y comparar el rendimiento de distintos algoritmos de *Machine Learning*** aplicados a la **detección de tráfico anómalo e intrusiones en redes**, utilizando el conjunto de datos **[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)**.

## Características principales

- **Carga de conjuntos de datos** en formato CSV (training y testing).
- **Preprocesamiento de datos**, incluyendo limpieza, normalización y selección de características.
- **Entrenamiento y evaluación de múltiples algoritmos supervisados**, tales como:
  - Árboles de Decisión
  - Bagging
  - Random Forest
- **Comparación de métricas de desempeño**: precisión, recall, F1-score, matriz de confusión, entre otras.
- **Visualización de resultados mediante gráficos y reportes** generados automáticamente.
- **Interfaz web amigable**, desarrollada con **Flask** y tecnologías web (HTML, CSS y JavaScript).

Este sistema busca demostrar cómo distintas técnicas de *Machine Learning* pueden aplicarse al ámbito de la **ciberseguridad**, ofreciendo una herramienta flexible para evaluar y analizar modelos en escenarios reales de detección de intrusiones.

## Tecnologías utilizadas

- **Python**: pandas, scikit-learn, matplotlib, xgboost
- **Flask**: backend y API
- **HTML, CSS, JavaScript**: frontend e interacción con usuarios
- **Render / GitHub Pages**: despliegue

## Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/usuario/nombre-repo.git
   cd nombre-repo
2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt

3. **Ejecutar la aplicacion:**
   ```bash
   python app.py

   
