import os
from PIL import Image
from tqdm import tqdm


def convert_png_to_jpg(directory):
    # Ensure directory ends with a separator
    if not directory.endswith(os.sep):
        directory += os.sep

    # List all files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.png'):
            # Open the image using PIL
            with Image.open(directory + file_name) as image:
                image = image.convert('RGB')
                jpg_name = os.path.splitext(file_name)[0] + '.jpg'
                image.save(directory + jpg_name, 'JPEG')

def remove_png_files(directory):
    # Ensure directory ends with a separator
    if not directory.endswith(os.sep):
        directory += os.sep

    # List all files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.png'):
            os.remove(directory + file_name)
    

if __name__ == '__main__':
    dir_path = input("Enter the directory path containing PNG images: ")
    convert_png_to_jpg(dir_path)
    print("Conversion complete!")
    remove_png_files(dir_path)
    print("PNG files removed!")
