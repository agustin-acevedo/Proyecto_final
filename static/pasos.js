/*
function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Por favor, selecciona un archivo primero.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Se envia el archivo al backend
    fetch('/upload', { 
        method: 'POST',
        body: formData,
    })
    .then(response => {
        if (response.ok) {
            return response.json(); // Traigo la ruta del archivo donde se guardo 
        } else {
            throw new Error('Error al subir el archivo.');
        }
    })
    .then(data => {
        console.log('Archivo subido con éxito:', data.filePath);
        alert('Archivo subido correctamente.');
        // Llama a goToStep para avanzar al preprocesamiento
        goToStep(3, data.filePath);
    })
    .catch(error => {
        console.error('Error en la solicitud:', error);
        alert('Hubo un problema al subir el archivo.');
    });
}


*/
/*
function goToStep(step) {
    if (step === 2) {
        // Obtener el archivo del input
        const fileInput = document.getElementById('file-input');
        const file = fileInput.files[0]; // Primer archivo seleccionado

        if (!file) {
            alert('Por favor, selecciona un archivo antes de continuar.');
            return;
        }

        // Crear un objeto FormData para enviar el archivo al servidor
        const formData = new FormData();
        formData.append('file', file);

        // Enviar el archivo al backend
        fetch('/upload', { // Cambia '/upload' a la ruta de tu backend
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (response.ok) {
                console.log('Archivo subido con éxito');
                // Cambiar al siguiente paso
                showStep(step, data.filePath);
            } else {
                console.error('Error al subir el archivo');
                alert('No se pudo subir el archivo. Inténtalo de nuevo.');
            }
        })
        .catch(error => {
            console.error('Error en la solicitud:', error);
            alert('Hubo un problema con la subida del archivo.');
        });
    } else {
        if (step === 3) {
            //LOGICA PARA CONECTAR CON EL BACKEND DEL PREPROCESAMIENTO 
            
        }
        // Para otros pasos, solo cambiar
        showStep(step);
    }
}
¨*/
function goToStep(step) {
    if (step === 2) {
        const fileInput = document.getElementById('file-input');
        const file = fileInput.files[0];

        if (!file) {
            alert('Por favor, selecciona un archivo primero.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        // Envía el archivo al backend
        fetch('/upload', { // Cambia '/upload' según tu endpoint
            method: 'POST',
            body: formData,
        })
            .then(response => {
                if (response.ok) {
                    return response.json(); // Espera la ruta del archivo en la respuesta
                } else {
                    throw new Error('Error al subir el archivo.');
                }
            })
            .then(data => {
                console.log('Archivo subido con éxito:', data.filePath);
                alert('Archivo subido correctamente.');
                // Llama a goToStep para avanzar al preprocesamiento
                //goToStep(3, data.filePath);

                ruta = data.filePath
            })
            .catch(error => {
                console.error('Error en la solicitud:', error);
                alert('Hubo un problema al subir el archivo.');
            });
    } else {
        if (step === 3) {
            //LOGICA PARA CONECTAR CON EL BACKEND DEL PREPROCESAMIENTO 
            preprocesarDatos(ruta, () => {
                // Avanza al siguiente paso después del preprocesamiento
                //showStep(4);
            });
        } else {
            if (step === 4) {
                trainModel();
            } else {
                if (step === 5) {
                    generarResultados()
                }
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



function preprocesarDatos(filePath, callback) {
    if (!filePath) {
        alert('No se encontró la ruta del archivo para procesar.');
        return;
    }

    // Muestra un mensaje al usuario mientras se procesa
    const messages = [
        "Empezando el procesamiento...",
        "Dividiendo el conjunto de datos...",
        "Aplicando transformaciones...",
        "Generando resultados y guardando...",
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
    fetch('/preprocess', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filePath }),
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

let globalModelo = modelSelect.value; // Lo que realizo es guardar el modelo seleccionado para usarlo en la funicon de generar grafica
function trainModel() {
    // Obtener el modelo seleccionado del <select>
    const modelSelect = document.getElementById('modelSelect');
    const selectedModel = modelSelect.value; // Captura el valor seleccionado

    if (!selectedModel) {
        alert('Por favor, selecciona un modelo.');
        return;
    }
    const messages = [
        "Empezando con el entrenamiento del modelo...",
        "Testeando el clasificador...",
        "Lectura del fichero...",
        "Empezando la fase de entrenamiento...",
        "Empezando la fase de clasificación...",
        "Obteniendo las metricas de rendimiento...",
        "Gruadando los resultados en el disco...",
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
    }, 4000);

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
            console.log('Preprocesamiento completado:', data);
        })
        .catch(error => {
            console.error('Hubo un problema:', error);
            alert('Ocurrió un error durante el entrenamiento.');
            statusMessage.style.display = 'none';
        });
}

function generarResultados() {
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
            alert('Archivo subido correctamente.');
            // Llama a goToStep para avanzar al preprocesamiento
            //goToStep(3, data.filePath);

            //ruta = data.filePath
        })
        .catch(error => {
            console.error('Error en la solicitud:', error);
            alert('Hubo un problema al subir el archivo.');
        });
}

