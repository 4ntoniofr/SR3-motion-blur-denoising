from PIL import Image
from tqdm import tqdm
import libnoise
import os
import matplotlib.image as mpimg
import numpy as np
import random

random.seed(42)


def min_max_norm(img):
    min_val = np.min(img)
    max_val = np.max(img)
    return (img - min_val) / (max_val - min_val)


def add_motion_blur_dir(angle_array, input_dir, length_array, output_dir_array):
    time = 1
    # Get a list of all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    # Process each image file
    for image_file in tqdm(image_files, desc="Añadiendo ruido por movimiento", unit="imágenes"):
        # Read the image
        try:
            img = mpimg.imread(os.path.join(input_dir, image_file))
        except OSError:
            os.remove(os.path.join(input_dir, image_file))
            continue
        # Apply transformation (for demonstration, let's take the negative of the image)
        selected_length = random.randint(0, len(length_array) - 1)
        selected_angle = random.randint(0, len(angle_array) - 1)

        blurred_img, _ = libnoise.get_mb_image(img, time,
                                               [[length_array[selected_length], angle_array[selected_angle]], [0, 0]])
        normalized_blurred_img = min_max_norm(blurred_img)

        image_int = (normalized_blurred_img * 255).astype(np.uint8)
        final_image = Image.fromarray(image_int)

        # Save the transformed image to the output directory
        for output_dir in output_dir_array:
            output_file = os.path.join(output_dir, image_file)
            final_image.save(output_file)


def main():
    dataset_dir = input(
        "Ingrese el directorio del conjunto de datos (predeterminado: SR3/dataset/train_camus): ") or 'SR3/dataset/train_camus'

    input_dir = os.path.join(dataset_dir, [f for f in os.listdir(dataset_dir) if f.startswith('hr_')][0])
    output_dirs = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if
                   f.startswith('lr_') or f.startswith('sr_')]

    length_array = [float(x) for x in
                    input(
                        "Ingrese números decimales separados por espacios que representen todas las longitudes posibles: ").split()]
    angle_array = [int(x) for x in input(
        "Ingrese números decimales separados por espacios que representen todos los ángulos posibles: ").split()]

    add_motion_blur_dir(angle_array, input_dir, length_array, output_dirs)

    print("Ruido por movimiento añadido al conjunto de datos satisfactoriamente.")


if __name__ == "__main__":
    main()
