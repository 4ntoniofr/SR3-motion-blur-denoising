import subprocess
import os
import skimage.io as io
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
from multiprocessing import Process
import json


def calculate_ssim(image1_path, image2_path):
    try:
        image1 = io.imread(image1_path, as_gray=True)
        image2 = io.imread(image2_path, as_gray=True)
        return ssim(image1, image2, data_range=image2.max() - image2.min())
    except Exception as e:
        print(f"Error calculating SSIM for {image1_path} and {image2_path}: {e}")
        return None


def calculate_mean_ssim(directory_path):
    ssim_values = {}
    for filename in os.listdir(directory_path):
        if filename.endswith("_hr.png"):
            hr_path = os.path.join(directory_path, filename)
            sr_filename = filename.replace("_hr.png", "_sr.png")
            sr_path = os.path.join(directory_path, sr_filename)
            if os.path.exists(sr_path):
                x_value = filename.split("_")[0]
                if x_value not in ssim_values:
                    ssim_values[x_value] = []
                ssim_value = calculate_ssim(hr_path, sr_path)
                if ssim_value is not None:
                    ssim_values[x_value].append(ssim_value)

    mean_ssim_values = {x: sum(ssim_list) / len(ssim_list) for x, ssim_list in ssim_values.items()}
    ssim_df = pd.DataFrame(mean_ssim_values.items(), columns=["X", "Mean SSIM"])
    ssim_df["X"] = ssim_df["X"].astype(int)
    ssim_df = ssim_df.sort_values("X")
    ssim_df.reset_index(drop=True, inplace=True)

    return ssim_df


def calculate_psnr(image1_path, image2_path):
    try:
        image1 = io.imread(image1_path)
        image2 = io.imread(image2_path)
        return psnr(image1, image2)
    except Exception as e:
        print(f"Error calculating PSNR for {image1_path} and {image2_path}: {e}")
        return None


def calculate_mean_psnr(directory_path):
    psnr_values = {}
    for filename in os.listdir(directory_path):
        if filename.endswith("_hr.png"):
            hr_path = os.path.join(directory_path, filename)
            sr_filename = filename.replace("_hr.png", "_sr.png")
            sr_path = os.path.join(directory_path, sr_filename)
            if os.path.exists(sr_path):
                x_value = filename.split("_")[0]
                if x_value not in psnr_values:
                    psnr_values[x_value] = []
                psnr_value = calculate_psnr(hr_path, sr_path)
                if psnr_value is not None:
                    psnr_values[x_value].append(psnr_value)

    mean_psnr_values = {x: sum(psnr_list) / len(psnr_list) for x, psnr_list in psnr_values.items()}
    psnr_df = pd.DataFrame(mean_psnr_values.items(), columns=["X", "Mean PSNR"])
    psnr_df["X"] = psnr_df["X"].astype(int)
    psnr_df = psnr_df.sort_values("X")
    psnr_df.reset_index(drop=True, inplace=True)

    return psnr_df


def obtener_metricas(input_directory, output_directory, resolution=128):
    print("Obteniendo metricas...")
    input_directory = os.path.join(input_directory, "hr_" + str(resolution))
    output_directory = os.path.join(output_directory, "results")
    while len(os.listdir(output_directory)) == 0:
        time.sleep(2)
        os.system("clear")
        print("Esperando a que se procesen las imágenes...")
    while len(os.listdir(output_directory)) * 5 < len(os.listdir(input_directory)):
        time.sleep(5)
        os.system("clear")
        mean_ssim = calculate_mean_ssim(output_directory)
        mean_psnr = calculate_mean_psnr(output_directory)
        processed_percentage = len(os.listdir(output_directory)) * 5 / len(os.listdir(input_directory)) * 100
        print(f"Directorio de resultados: {output_directory}")
        print(f"{processed_percentage:.4g}% de las imágenes procesadas")
        mean_ssim_value = float(mean_ssim['Mean SSIM'].iloc[0])
        mean_psnr_value = float(mean_psnr['Mean PSNR'].iloc[0])
        print(f"Media de SSIM: {mean_ssim_value:.4f}")
        print(f"Media de PSNR: {mean_psnr_value:.4f}")


def evaluar_SR3(config_file="eval_deblurring.json"):
    command = "python " + os.path.join("SR3", "sr.py") + " --config " + os.path.join("SR3", "config",
                                                                                     config_file) + " -p val"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process.wait()


def modificar_JSON_configuracion(config_file="eval_deblurring.json",
                                 pretrained_model="/SR3/pretrained_models/pretrained_motion_blur_camus/I580000_E180",
                                 input_directory="/SR3/dataset/test_camus",
                                 resolution=128):
    json_file = str(os.path.join("SR3", "config", config_file))

    with open(json_file, 'r') as file:
        data = json.load(file)

    data["path"]["resume_state"] = pretrained_model
    data["datasets"]["val"]["dataroot"] = input_directory
    data["datasets"]["val"]["l_resolution"] = resolution
    data["datasets"]["val"]["r_resolution"] = resolution

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)


def main():
    config_file = "eval_deblurring.json"
    pretrained_model = input(
        "Ingrese la ruta del modelo preentrenado (predeterminado: ./SR3/pretrained_models/pretrained_motion_blur_camus/I580000_E180): ") or "./SR3/pretrained_models/pretrained_motion_blur_camus/I580000_E180"
    input_directory = input(
        "Ingrese el conjunto de datos a evaluar en la carpeta SR3/datasets (predeterminado: ./SR3/dataset/test_camus): ") or "./SR3/dataset/test_camus"

    resolution_input = input("Ingrese la resolución de las imágenes (predeterminado: 128): ")
    resolution = int(resolution_input) if resolution_input else 128

    modificar_JSON_configuracion(config_file, pretrained_model, input_directory, resolution)

    os.makedirs("./experiments", exist_ok=True)
    experiments_before = os.listdir("./experiments")

    process_1 = Process(target=evaluar_SR3)
    process_1.start()

    experiments_after = os.listdir("./experiments")
    experiment_directory = list(set(experiments_after) - set(experiments_before))

    while len(experiment_directory) == 0:
        time.sleep(2)
        experiments_after = os.listdir("./experiments")
        experiment_directory = list(set(experiments_after) - set(experiments_before))

    experiment_directory = experiment_directory[0]
    process_2 = Process(target=obtener_metricas,
                        args=(input_directory, os.path.join("./experiments", experiment_directory)))
    process_2.start()

    process_1.join()
    process_2.join()


if __name__ == "__main__":
    main()
