import json
import os
import subprocess


def modificar_JSON_configuracion(n_iter=600000, val_freq=5000, save_freq=1000, print_freq=10, resume_state="",
                                 train_dataset='./SR3/dataset/train_camus',
                                 val_dataset='./SR3/dataset/validation_camus', resolution=128):
    json_file = str(os.path.join("SR3", os.path.join("config", "train_wizard_deblurring.json")))

    with open(json_file, 'r') as file:
        data = json.load(file)

    data["train"]["n_iter"] = n_iter
    data["train"]["val_freq"] = val_freq
    data["train"]["save_checkpoint_freq"] = save_freq
    data["train"]["print_freq"] = print_freq
    data["path"]["resume_state"] = None if resume_state == "" else resume_state
    data["datasets"]["train"]["dataroot"] = train_dataset
    data["datasets"]["val"]["dataroot"] = val_dataset
    data["datasets"]["train"]["l_resolution"] = resolution
    data["datasets"]["train"]["r_resolution"] = resolution
    data["datasets"]["val"]["l_resolution"] = resolution
    data["datasets"]["val"]["r_resolution"] = resolution
    data["model"]["diffusion"]["image_size"] = resolution

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)


def entrenar_SR3():
    command = "python " + os.path.join("SR3", "sr.py") + " --config " + os.path.join("SR3", os.path.join("config",
                                                                                                         "train_wizard_deblurring.json")) + " -p train"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.stdout.read().decode("utf-8"))


def main():
    resume_state = input(
        "Ingrese la ruta del punto de control de entrenamiento para reanudar (predeterminado: ninguno): ") or ""
    train_dataset = input(
        "Ingrese el directorio para guardar imágenes de entrenamiento (predeterminado: './SR3/dataset/train_camus'): ") or './SR3/dataset/train_camus'
    val_dataset = input(
        "Ingrese el directorio para guardar imágenes de validación (predeterminado: './SR3/dataset/validation_camus'): ") or './SR3/dataset/validation_camus'
    resolution = input("Ingrese la resolución de las imágenes (predeterminado: 128): ") or 128
    n_iter = input(
        "Ingrese el número de iteraciones durante las que el modelo entrenará (predeterminado: 600000): ") or 600000
    val_freq = input("Ingrese la frecuencia de validación en iteraciones (predeterminado: 5000): ") or 5000
    save_freq = input(
        "Ingrese la frecuencia de guardado de puntos de control en iteraciones (predeterminado: 1000): ") or 1000
    print_freq = input("Ingrese la frecuencia de impresión de resultados en iteraciones (predeterminado: 10): ") or 10

    modificar_JSON_configuracion(n_iter, val_freq, save_freq, print_freq, resume_state, train_dataset, val_dataset,
                                 resolution)
    entrenar_SR3()


if __name__ == "__main__":
    main()
