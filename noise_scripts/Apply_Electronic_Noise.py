import os
import random
from PIL import Image
from tqdm import tqdm
import numpy as np

random.seed(42)


def add_electronic_noise(image_path, noise_level):
    # Load the image
    img = np.array(Image.open(image_path))

    # Add electronic noise
    noise = np.random.normal(0, noise_level, img.shape)
    noisy_image = img + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


def add_electronic_noise_dir(input_dir, noise_levels, output_dirs):
    # Get a list of all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for image_file in tqdm(image_files, desc="Añadiendo ruido electrónico", unit="imágenes"):
        image_path = os.path.join(input_dir, image_file)

        try:
            img = Image.open(image_path)
        except (OSError, IOError):
            os.remove(image_path)
            continue

        selected_noise_level = random.choice(noise_levels)
        noisy_img = add_electronic_noise(image_path, selected_noise_level)

        # Save the noisy image to the output directories
        for output_dir in output_dirs:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, image_file)
            Image.fromarray(noisy_img).save(output_file)


def main():
    dataset_dir = input(
        "Ingrese el directorio del conjunto de datos (predeterminado: SR3/dataset/train_camus): ") or 'SR3/dataset/train_camus'

    input_dir = os.path.join(dataset_dir, next(f for f in os.listdir(dataset_dir) if f.startswith('hr_')))
    output_dirs = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if
                   f.startswith('lr_') or f.startswith('sr_')]

    noise_levels = list(map(float, input("Ingrese niveles de ruido separados por espacios: ").split()))

    add_electronic_noise_dir(input_dir, noise_levels, output_dirs)
    print("Ruido electrónico añadido al conjunto de datos satisfactoriamente.")


if __name__ == "__main__":
    main()