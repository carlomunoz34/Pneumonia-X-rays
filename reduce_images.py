import numpy as np
import skimage as sk
from glob import glob
import sys
import cv2

image_size = 224

def reduce(path, save_path):
    path += '*'
    files = glob(path)
    
    print("Reducing images in", path, "Stored in", save_path)
    i = 1
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

        new_image_path = '%sreduced_%i.jpeg' %(save_path, i)
        sk.io.imsave(new_image_path, img)
        sys.stdout.write("\r" + str(i) + "/" + str(len(files)))
        i += 1

    sys.stdout.write("\n")

if __name__ == "__main__":
    types = ['train/', 'test/', 'val/']
    classes = ['PNEUMONIA/', 'NORMAL/']
    base_path = '../../Datasets/chest_xray/'
    save_base_path = '../../Datasets/chest_xray/small/%i/' %(image_size)

    for type in types:
        for current_class in classes:
            current_path = base_path + type + current_class
            current_save_path = save_base_path + type + current_class

            reduce(current_path, current_save_path)