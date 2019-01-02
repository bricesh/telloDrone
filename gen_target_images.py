import cv2
import numpy as np
import random

"""
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python
"""

def main():
    """
    Generate training images for object detection based on single image of target
    """

    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)])
        return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

    def adjust_darkness(image, darkness=0):
        return (np.double(image) + darkness).astype(np.uint8)

    def adjust_size(image, size=1):
        return cv2.resize(image,None,fx=size, fy=size, interpolation = cv2.INTER_CUBIC)

    def adjust_rotation(image, rot=0):
        rows,cols = image.shape[0:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
        return cv2.warpAffine(image,M,(cols,rows))

    source_image_path = './Pictures/target_new.jpg' #location of the image
    original = cv2.imread(source_image_path, 1)
    small = cv2.resize(original, (0,0), fx=0.1, fy=0.1)
    gen_dir_path = "./Pictures/gen_target/"

    image_number = 1
    while image_number < 100:
        gamma = random.sample(range(30, 101), 1)[0] / 100.0     # 0.1 to 1.0
        darkness = random.sample(range(-10, 11), 1)[0]          # -10 to +10
        size = random.sample(range(50, 151), 1)[0] / 100.0      # .5 to 1.5
        rot = random.sample(range(-100, 101), 1)[0] / 10.0      # -15 to 15
        # To-Do (potentially):
        #   -Blur
        #   -Perspective
        #   -random colour background for rotated image

        adjusted = adjust_gamma(small, gamma=gamma)    
        adjusted = adjust_darkness(adjusted, darkness=darkness)
        adjusted = adjust_size(adjusted, size=size)
        adjusted = adjust_rotation(adjusted, rot=rot)
        
        cv2.imwrite("{}image_{}_g_{}_d_{}_s_{}_r_{}.jpeg".format(gen_dir_path,image_number,gamma,darkness,size,rot), adjusted)
        image_number += 1

if __name__ == '__main__':
    main()