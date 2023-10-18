#
# Modified version of 
# https://git.scc.kit.edu/bz3385/entomoscope_2_0_software/-/blob/main/stacker.py?ref_type=heads
#
# modified by: Joshua Albiez
#

import os
import cv2
import numpy
import subprocess
from threading import Thread


class Stacker(Thread):
    _BLUR_SIZE = 1
    _SIGMAX = 0
    _KERNEL_SIZE = 31 # default 7
    
    def __init__(self, raw_img_dir_path: str, stacked_img_name: str) -> None:
        super().__init__()
        
        self.raw_img_dir_path = raw_img_dir_path
        self.stacked_img_name = stacked_img_name
        
    def run(self):
        print("Preparing to fuse stacks...")
        
        raw_img_list = []
        gray_img_list = []
        for img_name in os.listdir(self.raw_img_dir_path):
            img_path = os.path.join(self.raw_img_dir_path, img_name)
            image = cv2.imread(img_path)
            raw_img_list.append(image)
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_img_list.append(gray_image)

        aligned_images = self.align_images(gray_img_list)
        
        print("images aligned")
        stacked_img = self.focus_stack(raw_img_list, aligned_images)
        
        cv2.imwrite(self.stacked_img_name, stacked_img)
        
    def align_images(self, raw_img_list: list) -> list:
        """Alignes images"""
        
        aligned_imgs = [raw_img_list[0]]
        for i, img2 in enumerate(raw_img_list[1:]):
            print(f"aligning img {i+1}")
            img1 = aligned_imgs[i]
            
            sz = img1.shape
            warp_mode = cv2.MOTION_TRANSLATION
            warp_matrix = numpy.eye(2, 3, dtype=numpy.float32)

            number_of_iterations = 5000
            termination_eps = 1e-10
            criteria = (int(cv2.TERM_CRITERIA_EPS / cv2.TERM_CRITERIA_COUNT), number_of_iterations, termination_eps)
            
            _, warp_matrix = cv2.findTransformECC(img1, img2, warp_matrix, warp_mode, criteria)
            img2_aligned = cv2.warpAffine(img2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            
            aligned_imgs.append(img2_aligned)
        
        return aligned_imgs
        
    def focus_stack(self, images, gray_images):
        """
        Find the sharpest area of each of the superimposed images 
        and generates an image from the different sharp areas.
        """
        
        laplacians_img = []
        for raw_image in gray_images:
            blurred = cv2.GaussianBlur(raw_image, (self._BLUR_SIZE, self._BLUR_SIZE), self._SIGMAX)
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=self._KERNEL_SIZE)

            laplacians_img.append(laplacian)
        
        laplacians_img = numpy.asarray(laplacians_img)
        output = numpy.zeros(shape=images[0].shape, dtype=images[0].dtype)

        abs_laplacians_img = numpy.absolute(laplacians_img)
        maxima = abs_laplacians_img.max(axis=0)
        bool_mask = abs_laplacians_img == maxima
        mask = bool_mask.astype(numpy.uint8)
        
        for i, img in enumerate(images):
            output = cv2.bitwise_not(img, output, mask=mask[i])
        
        return 255 - output


def fuse_stacks(dir_path, img_name):

    # Fuse stacks with selfmade version
    stacker = Stacker(dir_path, img_name)
    stacker.start()

if __name__ == '__main__':
    dir_path = '/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/stack_imgs/HiDrive-CAS001_CAS0000001_RAW_Data_01'
    stacked_img_name = 'stacked_test1.jpg'

    fuse_stacks(dir_path, stacked_img_name)