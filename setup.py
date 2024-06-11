import requests
from tqdm import tqdm
import os
import nibabel as nib
from PIL import Image
import numpy as np
import zipfile
import random
import gdown


def descargar_dataset_CAMUS():
    url = 'https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/63fde55f73e9f004868fb7ac/download'
    gd_link = 'https://drive.google.com/uc?id=1TIl3BX_f456zNzWAOMSIRNkcssBWAqwm'

    try:
        response = requests.get(url, stream=True)

        # Verificar si la descarga fue exitosa
        if response.status_code == 200:
            # Crear una barra de carga en porcentaje
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1 MB
            with open('camus-dataset.zip', 'wb') as file:
                for data in tqdm(response.iter_content(block_size), total=total_size // block_size, unit='kB',
                                 unit_scale=True):
                    file.write(data)
            print('¡El conjunto de datos CAMUS ha sido descargado exitosamente!')
        else:
            raise Exception(f"Código de estado: {response.status_code}")
    except Exception as e:
        print(f"Error al descargar el conjunto de datos CAMUS desde la URL principal")
        print("Intentando descargar desde Google Drive...")
        try:
            gdown.download(gd_link, 'camus-dataset.zip', quiet=False)
            print('¡El conjunto de datos CAMUS ha sido descargado exitosamente desde Google Drive!')
        except Exception as gd_error:
            print(f"Error al descargar el conjunto de datos CAMUS desde Google Drive: {gd_error}")


def extraer_dataset_CAMUS(path_to_zip_file='camus-dataset.zip', directory_to_extract_to='./CAMUS_dataset/'):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Extrayendo conjunto de datos CAMUS", unit="archivos"):
            zip_ref.extract(file, directory_to_extract_to)


def listar_archivos_nii_recursivamente(directory):
    nii_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".nii.gz") and not file.endswith("_gt.nii.gz"):
                nii_files.append(os.path.join(root, file))
    return nii_files


def NIB_a_PNG(input_dir, train_percent=0.6, val_percent=0.2, test_percent=0.2, train_dir="./SR3/dataset/train_camus",
              val_dir="./SR3/dataset/validation_camus",
              test_dir="./SR3/dataset/test_camus"):
    # Crear directorios si no existen
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "hr_128"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "lr_128"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "sr_128_128"), exist_ok=True)

    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(val_dir, "hr_128"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "lr_128"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "sr_128_128"), exist_ok=True)

    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, "hr_128"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "lr_128"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "sr_128_128"), exist_ok=True)

    # Listar recursivamente todos los archivos nii.gz en el directorio de entrada, excluyendo los que terminan en _gt.nii.gz
    files = listar_archivos_nii_recursivamente(input_dir)

    # Barajar los archivos
    random.shuffle(files)

    # Calcular el número de archivos para cada conjunto
    total_files = len(files)
    train_count = int(total_files * train_percent)
    val_count = int(total_files * val_percent)
    test_count = total_files - train_count - val_count

    # Dividir los archivos en conjuntos de entrenamiento, validación y prueba
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    def procesar_y_guardar(file, output_dir, size):
        try:
            raw_img = nib.load(file)
            array = raw_img.get_fdata()
        except Exception as e:
            return

        if len(array.shape) == 3:
            for i in range(array.shape[-1]):
                image = Image.fromarray(np.uint8(array[:, :, i])).rotate(-90, expand=True).resize(size)
                filename = os.path.basename(file)[:-7] + "-" + str(i + 1) + ".png"
                image.save(os.path.join(output_dir, os.path.join("hr_128", filename)))
                image.save(os.path.join(output_dir, os.path.join("lr_128", filename)))
                image.save(os.path.join(output_dir, os.path.join("sr_128_128", filename)))
        elif len(array.shape) == 2:
            image = Image.fromarray(np.uint8(array)).rotate(-90, expand=True).resize(size)
            filename = os.path.basename(file)[:-7] + ".png"
            image.save(os.path.join(output_dir, os.path.join("hr_128", filename)))
            image.save(os.path.join(output_dir, os.path.join("lr_128", filename)))
            image.save(os.path.join(output_dir, os.path.join("sr_128_128", filename)))
        else:
            print(f"Error en la imagen {file}")

    # Procesar y guardar archivos en los directorios correspondientes
    for file in tqdm(train_files, desc="Procesando archivos de entrenamiento", unit="archivos"):
        procesar_y_guardar(file, train_dir, (128, 128))

    for file in tqdm(val_files, desc="Procesando archivos de validación", unit="archivos"):
        procesar_y_guardar(file, val_dir, (128, 128))

    for file in tqdm(test_files, desc="Procesando archivos de prueba", unit="archivos"):
        procesar_y_guardar(file, test_dir, (128, 128))


def main():
    already_downloaded = input("¿Ya descargó el conjunto de datos CAMUS? (s/N): ") or 'N'
    input_dir = "./CAMUS_dataset/"
    if already_downloaded.lower() != 's':
        print("Descargando conjunto de datos CAMUS...")
        descargar_dataset_CAMUS()
        print("Extrayendo conjunto de datos CAMUS...")
        extraer_dataset_CAMUS()
    else:
        input_dir = input(
            "Ingrese el directorio donde se encuentra el conjunto de datos CAMUS extraído (predeterminado: './CAMUS_dataset/'): ") or './CAMUS_dataset/'

    train_dir = input(
        "Ingrese el directorio para guardar imágenes de entrenamiento (predeterminado: './SR3/dataset/train_camus'): ") or './SR3/dataset/train_camus'
    val_dir = input(
        "Ingrese el directorio para guardar imágenes de validación (predeterminado: './SR3/dataset/validation_camus'): ") or './SR3/dataset/validation_camus'
    test_dir = input(
        "Ingrese el directorio para guardar imágenes de prueba (predeterminado: './SR3/dataset/test_camus'): ") or './SR3/dataset/test_camus'
    train_percent_input = input("Ingrese el porcentaje de datos para entrenamiento (predeterminado: 0.6): ")
    train_percent = float(train_percent_input) if train_percent_input else 0.6

    val_percent_input = input("Ingrese el porcentaje de datos para validación (predeterminado: 0.2): ")
    val_percent = float(val_percent_input) if val_percent_input else 0.2

    test_percent_input = input("Ingrese el porcentaje de datos para prueba (predeterminado: 0.2): ")
    test_percent = float(test_percent_input) if test_percent_input else 0.2

    # Asegurarse de que los porcentajes sumen 1
    train_percent = float(train_percent)
    val_percent = float(val_percent)
    test_percent = float(test_percent)
    if train_percent + val_percent + test_percent != 1.0:
        print("Error: La suma de los porcentajes de entrenamiento, validación y prueba debe ser 1.0")
        return

    print("Convirtiendo imágenes NIfTI a PNG...")
    NIB_a_PNG(input_dir, train_percent, val_percent, test_percent, train_dir, val_dir, test_dir)

    print("¡Procesamiento completado!")


if __name__ == "__main__":
    main()
