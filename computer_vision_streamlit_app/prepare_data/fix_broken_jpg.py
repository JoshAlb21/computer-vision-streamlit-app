# This script is a copy of 
# https://github.com/Poulinakis-Konstantinos/ML-util-functions/blob/master/scripts/Img_Premature_Ending-Detect_Fix.py
# Credits to Poulinakis-Konstantinos !

''' This script will detect and also fix the problem of Premature Ending in images.
This is caused when the image is corrupted in such a way that their hex code does not end with the default
D9. Opening the image with opencv and other image libraries is usually still possible, but the images might
produce errors during DL training or other tasks.
  Loading such an image with opencv and then saving it again can solve the problem. You can manually inspect 
,using a notepad, that the image's hex finishes with D9 after the script has finished. 
'''

import os 
import cv2 
from PIL import Image


def detect_and_fix(img_path, img_name):
    # detect for premature ending
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

def detect_and_fix_dir(dir_path:str, second_check:bool=False):

    for path in os.listdir(dir_path):
        # Make sure to change the extension if it is nor 'jpg' ( for example 'JPG','PNG' etc..)
        if path.endswith('.jpg'):
            img_path = os.path.join(dir_path, path)
            detect_and_fix(img_path=img_path, img_name = path)

        if second_check:
            im = Image.open(img_path)
            im.verify() #I perform also verify, don't know if he sees other types o defects
            im.close() #reload is necessary in my case
            im = Image.open(img_path) 
            im.transpose(Image.FLIP_LEFT_RIGHT)
            im.close()

    print("Process Finished")