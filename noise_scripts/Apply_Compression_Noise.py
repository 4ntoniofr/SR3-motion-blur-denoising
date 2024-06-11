import os
import random
from PIL import Image
from tqdm import tqdm
import matplotlib.image as mpimg

random.seed(42)


def add_compression_artifacts(image_path, quality):
    # Load the image
    img = Image.open(image_path)

    # Simulate compression by saving with lower quality
    temp_path = './temp_compressed.jpg'
    img.save(temp_path, quality=quality)
    img_compressed = Image.open(temp_path)

    # Return original and compressed images
    return img, img_compressed


def add_compression_artifacts_dir(input_dir, output_dirs, quality_array):
    # Get a list of all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    # Process each image file
    for image_file in tqdm(image_files, desc="Añadiendo artefactos de compresión", unit="imágenes"):
        image_path = os.path.join(input_dir, image_file)
        try:
            img = mpimg.imread(image_path)
        except (OSError, IOError):
            os.remove(image_path)
            continue

        selected_quality = random.choice(quality_array)
        _, compressed_img = add_compression_artifacts(image_path, selected_quality)

        # Save the transformed image to all output directories
        for output_dir in output_dirs:
            output_file = os.path.join(output_dir, image_file)
            compressed_img.save(output_file)


def main():
    dataset_dir = input(
        "Ingrese el directorio del conjunto de datos (predeterminado: SR3/dataset/train_camus): ") or 'SR3/dataset/train_camus'
    quality_array = list(
        map(int, input(
            "Ingrese números enteros separados por espacios que representen todas las calidades de compresión a establecer: ").split()))

    input_dir = os.path.join(dataset_dir, [f for f in os.listdir(dataset_dir) if f.startswith('hr_')][0])
    output_dirs = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if
                   f.startswith('lr_') or f.startswith('sr_')]

    add_compression_artifacts_dir(input_dir, output_dirs, quality_array)

    print("Ruido por compresión añadido al conjunto de datos satisfactoriamente.")


if __name__ == "__main__":
    main()
