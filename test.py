from PIL import Image
import os
import cv2

img_path = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/img/ver1/CAS002_CAS0000115_stacked_02_H.jpg"


img_name = os.path.basename(img_path)
im = Image.open(img_path)
im.verify() #I perform also verify, don't know if he sees other types o defects
im.close() #reload is necessary in my case
im = Image.open(img_path) 
im.transpose(Image.FLIP_LEFT_RIGHT)
im.close()

try:
    with open( img_path, 'rb') as im :
        im.seek(-2,2)
        if im.read() == b'\xff\xd9':
            print('Image OK :', img_name) 
        else: 
            # fix image
            img = cv2.imread(img_path)
            cv2.imwrite( img_path, img)
            print('FIXED corrupted image :', img_name)           
except(IOError, SyntaxError) as e :
    print(e)
    print("Unable to load/write Image : {} . Image might be destroyed".format(img_path) )


#YOLO check
# verify images
im = Image.open(img_path)
im.verify()  # PIL verify
if im.format.lower() in ('jpg', 'jpeg'):
    with open(img_path, 'rb') as f:
        f.seek(-2, 2)
        assert f.read() == b'\xff\xd9', 'corrupted JPEG'


from PIL import Image

def check_jpeg_integrity(file_path):
    try:
        with Image.open(file_path) as img:
            # Attempt to read the image
            img.load()
        return True
    except Exception as e:
        print(f"Image {file_path} is corrupted. Error: {e}")
        return False

# Test
if check_jpeg_integrity(img_path):
    print(f"Image {img_path} is not corrupted.")
else:
    print(f"Image {img_path} is corrupted.")


def check_jpeg_eof(file_path):
    with open(file_path, 'rb') as f:
        f.seek(-3, 2)  # Move to the 3rd last byte of the file
        last_three_bytes = f.read(3)
        
    if last_three_bytes[1:] == b'\xFF\xD9':
        return last_three_bytes[0]
    else:
        return None

extraneous_byte = check_jpeg_eof(img_path)

if extraneous_byte is not None:
    print(f"The image {img_path} has {extraneous_byte} as an extraneous byte before the JPEG end-of-file marker.")
else:
    print(f"The image {img_path} does NOT have any extraneous bytes before the JPEG end-of-file marker.")
