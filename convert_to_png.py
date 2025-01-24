import os
from PIL import Image


def convert_images_to_png(input_dir, output_dir, dapi=False):
    """
    Convert all image files in the input directory to PNG format.

    Args:
    input_dir (str): Path to the directory containing source images
    output_dir (str): Path to the directory where PNG files will be saved
    dapi (bool): If True, only convert images with 'DAPI' in filename
    """
    os.makedirs(output_dir, exist_ok=True)
    supported_extensions = ['.tif', '.tiff', '.jpg', '.jpeg', '.bmp', '.gif']

    for filename in os.listdir(input_dir):
        if (any(filename.lower().endswith(ext) for ext in supported_extensions) and
                (not dapi or 'DAPI' in filename)):

            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_dir, output_filename)

            try:
                with Image.open(input_path) as img:
                    img.save(output_path)
                print(f"Converted: {filename} -> {output_filename}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")


if __name__ == '__main__':
    input_directory = 'images/raw'
    output_directory = 'images/converted'
    convert_images_to_png(input_directory, output_directory, dapi=True)