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
    for directory in os.listdir(directory_path):
        for filename in os.listdir(os.path.join(directory_path, directory)):
            if filename.endswith("_hr.png"):
                hr_path = os.path.join(directory_path, directory, filename)
                sr_filename = filename.replace("_hr.png", "_sr.png")
                sr_path = os.path.join(directory_path, directory, sr_filename)
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
    for directory in os.listdir(directory_path):
        for filename in os.listdir(os.path.join(directory_path, directory)):
            if filename.endswith("_hr.png"):
                hr_path = os.path.join(directory_path, directory, filename)
                sr_filename = filename.replace("_hr.png", "_sr.png")
                sr_path = os.path.join(directory_path, directory, sr_filename)
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


def obtener_metricas(input_directory, output_directory):
    while len(os.listdir(output_directory)) * 5 < len(os.listdir(input_directory)):
        time.sleep(30)
        os.system("clear")
        mean_ssim = calculate_mean_ssim(output_directory)
        mean_psnr = calculate_mean_psnr(output_directory)
        print(f"{len(os.listdir(output_directory)) * 5 / len(os.listdir(input_directory)) * 100}% of images processed")
        print(f"Mean SSIM values: {mean_ssim}")
        print(f"Mean PSNR values: {mean_psnr}")


def evaluar_SR3(config_file="eval_deblurring.json"):
    command = "python " + os.path.join("SR3", "sr.py") + " --config " + os.path.join("config",
                                                                                     config_file) + " -p eval"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.stdout.read().decode("utf-8"))


def modificar_JSON_configuracion(config_file="eval_deblurring.json",
                                 pretrained_model="/SR3/pretrained_models/pretrained_motion_blur_camus/I580000_E180",
                                 input_directory="/SR3/validation_camus"):
    json_file = str(os.path.join("SR3", os.path.join("config", config_file)))

    with open(json_file, 'r') as file:
        data = json.load(file)

    data["path"]["resume_state"] = pretrained_model
    data["datasets"]["val"]["dataroot"] = input_directory

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)


def main():
    config_file = input(
        "Ingrese el nombre del archivo de configuración (predeterminado: eval_deblurring.json): ") or "eval_deblurring.json"
    pretrained_model = input(
        "Ingrese la ruta del modelo preentrenado (predeterminado: /SR3/pretrained_models/pretrained_motion_blur_camus/I580000_E180): ") or "/SR3/pretrained_models/pretrained_motion_blur_camus/I580000_E180"
    input_directory = input(
        "Ingrese el conjunto de datos a evaluar en la carpeta SR3/datasets (predeterminado: /SR3/validation_camus): ") or "/SR3/validation_camus"

    modificar_JSON_configuracion(config_file, pretrained_model, input_directory)

    os.makedirs("experiments", exist_ok=True)
    experiments_before = os.listdir("experiments")

    process_1 = Process(target=evaluar_SR3, args=config_file)
    process_1.start()
    process_1.join()

    experiments_after = os.listdir("experiments")
    experiment_directory = list(set(experiments_after) - set(experiments_before))[0]

    while experiment_directory is None:
        time.sleep(30)
        experiments_after = os.listdir("experiments")
        experiment_directory = list(set(experiments_after) - set(experiments_before))[0]

    process_2 = Process(target=obtener_metricas, args=(
        os.path.join("SR3", "dataset", input_directory), os.path.join("SR3", "results", experiment_directory)))
    process_2.start()
    process_2.join()


if __name__ == "__main__":
    main()
