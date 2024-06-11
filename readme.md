# Eliminación de ruido por movimiento mediante refinamiento iterativo.

Este proyecto utiliza la aproximación del refinamiento iterativo, normalmente usada para la tarea de súper resolución (
SR3), para eliminar ruido por movimiento (entre otros) de diversas imágenes. En concreto se ha estudiado su aplicación a
imágenes ecocardiográficas.

## Uso

### Entorno

```bash
pip install -r requirement.txt
```

### Archivos principales

- **SR3/config:** directorio en el que se encuentran los archivos de configuración del modelo a entrenar o evaluar
- **noise_scripts:** directorio en el que se encuentran los scripts para añadir ruido por movimiento, ruido electrónico
  o ruido por compresión a las imágenes para poder entrenar y validar los modelos.
- **setup.py:** script para descargar, descomprimir y preparar el conjunto de datos con imágenes ecocardiográficas
  CAMUS.
- **train_wizard.py:** script para entrenar el modelo estudiado en el trabajo sin necesidad de modificar archivos de
  configuración con una capacidad de personalización limitada.
- **train.py:** script para entrenar un modelo de refinamiento iterativo con un archivo de configuración propio para una
  capacidad de personalización completa.
- **eval.py:** script para evaluar un modelo ya entrenado sobre un conjunto de datos y obtener las métricas PSNR y SSIM.

### Modelos pre-entrenados

Para seguir entrenando o evaluar modelos pre-entrenados ante distintos tipos de ruido como el ruido por movimiento, el
ruido electrónico o el ruido por compresión, se pueden descargar
de [este enlace](https://drive.google.com/file/d/1-o1SUInzfGPrsXc8ud6EIK3RABKnBLLL/view?usp=sharing) de Google Drive.

### Evaluar un modelo

```bash
>>> python eval.py

Ingrese la ruta del modelo preentrenado (predeterminado: ./SR3/pretrained_models/pretrained_motion_blur_camus/I580000_E180): 
Ingrese el conjunto de datos a evaluar en la carpeta SR3/datasets (predeterminado: ./SR3/dataset/test_camus):
Ingrese la resolución de las imágenes (predeterminado: 128):
```

Para evaluar modelos personalizados, modificar SR3/config/eval_deblurring.py. La configuración de la arquitectura del
modelo debe ser la misma en el archivo de configuración usado en el entrenamiento y en el usado en la validación.

### Preparar datos

#### CAMUS

El script setup.py descarga y prepara las imágenes del conjunto de datos CAMUS. Este contiene imágenes de
ecocardiografías de 128x128 píxeles.

```bash
>>> python setup.py

¿Ya descargó el conjunto de datos CAMUS? (s/N):
Descargando conjunto de datos CAMUS...
Extrayendo conjunto de datos CAMUS...
Ingrese el directorio para guardar imágenes de entrenamiento (predeterminado: './SR3/dataset/train_camus'):
Ingrese el directorio para guardar imágenes de validación (predeterminado: './SR3/dataset/validation_camus'):
Ingrese el directorio para guardar imágenes de prueba (predeterminado: './SR3/dataset/test_camus'):
Ingrese el porcentaje de datos para entrenamiento (predeterminado: 0.6)
Ingrese el porcentaje de datos para validación (predeterminado: 0.2)
Ingrese el porcentaje de datos para prueba (predeterminado: 0.2)
Convirtiendo imágenes NIfTI a PNG...
¡Procesamiento completado!

```

Esto creará en la carpeta SR3/dataset/ los conjuntos de datos train_camus, validation_camus y test_camus
correspondientes al conjunto de entrenamiento, validación y prueba respectivamente. Las imágenes se habrán repartido de
acuerdo a la proporción indicada durante la ejecución del script.

#### Conjunto de datos propio

Deberemos ejecutar el siguiente script para los conjuntos de entrenamiento, validación y prueba por separado.

```bash
>>> python SR3/data/prepare_data.py --path <dirección del conjunto de datos> --out SR3/dataset/<nombre del conjunto de datos> 
    --size <resolución de las imagenes>,<resolución de las imagenes>

```

Esto creará en la carpeta SR3/dataset/ una carpeta con el nombre dado, con las subcarpetas necesarias y con las imágenes
para el entrenamiento o validación de los modelos sobre el conjunto de datos introducido.

#### Añadir ruido

En el directorio /noise_scripts/ se encuentran los scripts que usaremos para añadir ruido a los conjuntos de datos
preparados en el paso anterior.

```bash
>>> python noise_scripts/Apply_Motion_Blur.py

Ingrese el directorio del conjunto de datos (predeterminado: SR3/dataset/train_camus):
Ingrese números decimales separados por espacios que representen todas las longitudes posibles: 0.01 0.03 0.05
Introduzca números decimales separados por espacios que representen todos los ángulos posibles: 0 15 30 45 60 75 90
Añadiendo ruido por movimiento...

```

### Entrenar un modelo

Para un entrenamiento básico:

```bash
>>> python train_wizard.py

Ingrese la ruta del punto de control de entrenamiento para reanudar (predeterminado: ninguno): 
Ingrese el directorio para guardar imágenes de entrenamiento (predeterminado: './SR3/dataset/train_camus'):
Ingrese el directorio para guardar imágenes de validación (predeterminado: './SR3/dataset/validation_camus'):
Ingrese la resolución de las imágenes (predeterminado: 128):
Ingrese el número de iteraciones durante las que el modelo entrenará (predeterminado: 600000): 
Ingrese la frecuencia de validación en iteraciones (predeterminado: 5000):
Ingrese la frecuencia de guardado de puntos de control en iteraciones (predeterminado: 10000):
Ingrese la frecuencia de impresión de resultados en iteraciones (predeterminado: 10):

```

Para un entrenamiento más avanzado y personalizado, modifica el archivo /config/train_deblurring.py.

```bash
>>> python train.py

Ingrese la ruta del archivo de configuración (predeterminado: ./SR3/config/train_deblurring.py):
```

## Referencias

- [1] [Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636)
- [2] [Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D Echocardiography](https://pubmed.ncbi.nlm.nih.gov/30802851/)
