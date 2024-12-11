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
           // goToStep(3, data.filePath);
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
                showStep(4);
            });
        }
    }
    // Para otros pasos, solo cambiar
    //showStep(step);
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
    alert('Iniciando preprocesamiento de datos...');

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
        console.log('Preprocesamiento completado:', data);
        alert('Preprocesamiento completado con éxito.');
        if (callback) callback();
    })
    .catch(error => {
        console.error('Error en la solicitud:', error);
        alert('Hubo un problema durante el preprocesamiento.');
    });
}



/*
function goToStep(stepNumber) {
    // Validar datos del paso 1
    if (stepNumber === 2) {
       const fileInput = document.getElementById('file-input');
       /*const formData = new FormData(document.getElementById('file-input'));
        if (!fileInput.value) {
            alert('Por favor, selecciona un archivo.');
            return;
        }else{
            fetch('/uploads', { method: 'POST', body: fileInput })
            .then(response => response.json())
            .then(data => {
                if (data.columns) {
                    document.getElementById("columns").innerHTML = "Columnas disponibles: " + data.columns.join(", ");
                    const select = document.getElementById("target");
                    select.innerHTML = data.columns.map(col => `<option value="${col}">${col}</option>`).join("");
                    document.getElementById("trainForm").style.display = "block";
                }
            });
        }
    }



    // Cambiar de paso
    const steps = document.querySelectorAll('.step');
    steps.forEach(step => step.classList.remove('active'));
    const currentStep = document.getElementById(`step-${stepNumber}`);
    currentStep.classList.add('active');
}
*/