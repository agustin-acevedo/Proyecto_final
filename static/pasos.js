let ruta 
let rutaTest 


document.addEventListener("DOMContentLoaded", () => {
    fetch("/uploads/test.csv", { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                showStep(2);
            } else {
                console.log("El archivo test.csv no existe");
            }
        })
        .catch(error => {
            console.error("Error al verificar el archivo:", error);
        });
});

function goToStep(step) {
    if (step === 2) {
        const trainFile = document.getElementById('trainFile').files[0];
        const testFile = document.getElementById('testFile').files[0];
    
        if (!trainFile || !testFile) {
            alert("Por favor, selecciona ambos archivos.");
            return;
        }
    
        const formData = new FormData();
        formData.append('trainFile', trainFile);
        formData.append('testFile', testFile);
    
        fetch('/upload-datasets', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log("Archivos subidos correctamente");
               // goToStep(3); // Ir al siguiente paso
                let ruta = data.filePath
                let rutaTest = data.testPath
                 //LOGICA PARA CONECTAR CON EL BACKEND DEL PREPROCESAMIENTO 
                preprocesarDatos(ruta, rutaTest, () => {
                    // Avanza al siguiente paso después del preprocesamiento
                    //showStep(4);
                });
            } else {
                alert("Error al subir archivos: " + data.error);
            }
        })
        .catch(error => {
            console.error("Error en la carga:", error);
        });
    } else {
        //if (step === 3) {
            //LOGICA PARA CONECTAR CON EL BACKEND DEL PREPROCESAMIENTO 
          //  preprocesarDatos(ruta, rutaTest, () => {
                // Avanza al siguiente paso después del preprocesamiento
                //showStep(4);
            //});

        //} else {
            if (step === 3) {
                trainModel();
            } else {
                if (step === 4) {
                    generarResultados()
                }
            }
        }
    
    // Para otros pasos, solo cambiar
    showStep(step);
}


// Mostrar el paso actual y ocultar los demás
function showStep(step) {
    const steps = document.querySelectorAll('.step');
    steps.forEach((stepDiv, index) => {
        stepDiv.classList.toggle('active', index + 1 === step);
    });
}



function preprocesarDatos(filePath,rutaTest, callback) {
    if (!filePath || !rutaTest) {
        alert('No se encontró la ruta del archivo para procesar.');
        return;
    }

    // Muestra un mensaje al usuario mientras se procesa
    const messages = [
        "Empezando el preprocesamiento...",
        "Procesando los conjuntos de datos...",
        "Aplicando transformaciones...",
        "Generando resultados...",
        "Guardando datos preprocesados",
        "Ya casi esta listo..."
    ];

    let index = 0;
    const loadingText = document.getElementById("loading-text");
    const loadingMessage = document.getElementById("loading-message");

    // Mostrar el overlay de carga
    loadingMessage.style.display = "flex";

    // Actualizar el mensaje cada 4 segundos
    const intervalId = setInterval(() => {
        if (index < messages.length) {
            loadingText.textContent = messages[index];
            index++;
        } else {
            clearInterval(intervalId);
        }
    }, 4000);
    // Envía la ruta del archivo al backend
    console.log("Preprocesar con:", ruta, rutaTest);
    fetch('/preprocess', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filePath, rutaTest }),
    })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Error durante el preprocesamiento.');
            }
        })
        .then(data => {
            //console.log('Preprocesamiento completado:', data);
            //alert('Preprocesamiento completado con éxito.');
            // Ocultar el overlay al completar
            clearInterval(intervalId);
            loadingMessage.style.display = "none";
            console.log('Preprocesamiento completado:', data);
            if (callback) callback();
        })
        .catch(error => {
            console.error('Error en la solicitud:', error);
            // alert('Hubo un problema durante el preprocesamiento.');
            // Ocultar el overlay al completar
            clearInterval(intervalId);
            loadingMessage.style.display = "none";
        });
}

//let modelSelect = document.getElementById('modelSelect');
//let globalModelo = modelSelect.value; // Lo que realizo es guardar el modelo seleccionado para usarlo en la funicon de generar grafica
function trainModel() {
    // Obtener el modelo seleccionado del <select>
    const modelSelect = document.getElementById('modelSelect');
    const selectedModel = modelSelect.value; // Captura el valor seleccionado

    if (!selectedModel) {
        alert('Por favor, selecciona un modelo.');
        return;
    }
    const messages = [
        "Iniciando...",
        "Obteniendo los datos preprocesados...",
        "Lectura del fichero...",
        "Empezando la fase de entrenamiento...",
        "Empezando la fase de clasificación...",
        "Obteniendo las metricas de rendimiento...",
        "Guardando los resultados en el disco...",
        "Ya casi esta listo, aguarde un momento..."
    ];

    let index = 0;
    const loadingText = document.getElementById("training-status");
    const loadingMessage = document.getElementById("loading-message-train");

    // Mostrar mensaje de estado
    // const statusMessage = document.getElementById('training-status');
    //statusMessage.style.display = 'block';
    //statusMessage.textContent = 'Entrenando modelo...';

    // Mostrar el overlay de carga
    loadingMessage.style.display = "flex";

    // Actualizar el mensaje cada 4 segundos
    const intervalId = setInterval(() => {
        if (index < messages.length) {
            loadingText.textContent = messages[index];
            index++;
        } else {
            clearInterval(intervalId);
        }
    }, 2000);

    // Enviar el modelo seleccionado al backend
    fetch('/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model: selectedModel })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Error durante el entrenamiento');
            }
            return response.json();
        })
        .then(data => {
            //console.log('Entrenamiento completado:', data);
            //alert(`Entrenamiento completado. Resultado: ${data.message}`);
            //statusMessage.style.display = 'none';
            clearInterval(intervalId);
            loadingMessage.style.display = "none";
            console.log('Entrenamiento completado:', data);
            alert('¡El modelo se entreno correctamente!');
        })
        .catch(error => {
            console.error('Hubo un problema:', error);
            alert('Ocurrió un error durante el entrenamiento.');
            statusMessage.style.display = 'none';
        });
}

function generarResultados() {

    const messages = [
        "Generando Graficos y tablas...",
        "Guardando los resultados...", 
        "Ya casi esta listo..."
    ];

    let index = 0;
    const loadingText = document.getElementById("grafica-status");
    const loadingMessage = document.getElementById("loading-message-grafica");


    // Mostrar el overlay de carga
    loadingMessage.style.display = "flex";

    // Actualizar el mensaje cada 4 segundos
    const intervalId = setInterval(() => {
        if (index < messages.length) {
            loadingText.textContent = messages[index];
            index++;
        } else {
            clearInterval(intervalId);
        }
    }, 2000);


    // Envía el archivo al backend
    fetch('/results', { 
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
          }
    })
        .then(response => {
            if (response.ok) {
                return response.json(); 
            } else {
                throw new Error('Error al generar las graficas.');
            }
        })
        .then(data => {
           // alert('¡Graficos generados correctamente!');
            // Llama a goToStep para avanzar al preprocesamiento
            //goToStep(3, data.filePath);

            //ruta = data.filePath
            clearInterval(intervalId);
            loadingMessage.style.display = "none";
            //console.log('Preprocesamiento completado:', data);
            alert('¡Graficos generados correctamente!');
        })
        .catch(error => {
            console.error('Error en la solicitud:', error);
            alert('Hubo un problema al generar las graficas.');
            statusMessage.style.display = 'none';
        });
}

