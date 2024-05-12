from PIL import Image
from tqdm import tqdm
import os

def convert_tif_to_x(source_img_path: str, target_img_path: str) -> None:

    if not target_img_path.endswith(".png") and not target_img_path.endswith(".jpg"):
        raise ValueError("Target image path must end with .png or .jpg")
    
    #open tiff image 
    im = Image.open(source_img_path)
    im.save(target_img_path)

if __name__ == '__main__':
    source_img_folder = '/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/stack_imgs/HiDrive-CAS001_CAS0000001_RAW_Data_01'
    target_img_folder = '/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/stack_imgs/HiDrive-CAS001_CAS0000001_RAW_Data_01/png'

    # List all images in the data folder
    image_names = [img for img in os.listdir(source_img_folder) if img.endswith(('tif', 'tiff'))]

    # Create target folder if it does not exist
    if not os.path.exists(target_img_folder):
        os.makedirs(target_img_folder)

    print("Start converting images from tif to png...")
    for image in tqdm(image_names):
        print("Converting image", image)
        source_img_path = os.path.join(source_img_folder, image)
        target_img_path = os.path.join(target_img_folder, f'{image.split(".")[0]}.png')
        convert_tif_to_x(source_img_path, target_img_path)
    print("Finished converting images from tif to png.")