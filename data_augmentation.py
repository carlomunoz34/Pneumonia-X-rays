import numpy as np
import skimage as sk
from glob import glob
import sys

def random_rotation(image: np.ndarray):
    random_degree = np.random.uniform(-25, 25)
    return sk.transform.rotate(image, random_degree) * 255

def random_noise(image: np.ndarray):
    return sk.util.random_noise(image) * 255

def horizontal_flip(image: np.ndarray):
    return image[:, ::-1]

def transform(num_of_transformations, path):
    transform_list = [random_rotation, random_noise, horizontal_flip]
    
    #Augment pneumonia images
    files = glob(path + "*.jpeg")

    transformed_images = 0
    while transformed_images < num_of_transformations:
        #Choice the image to transform
        random_image = np.random.choice(files)
        image_to_transform = sk.io.imread(random_image)
        
        #apply the transformations
        num_transforms = np.random.randint(len(transform_list)) + 1

        for _ in range(num_transforms):
            current_transform = np.random.choice(transform_list)
            new_image = current_transform(image_to_transform).astype(np.uint8)
            
            transformed_images += 1

            #Save the new image
            new_image_path = '%saugmented_image_%i.jpg' %(path, transformed_images)
            sk.io.imsave(new_image_path, new_image)

            sys.stdout.write("\r" + str(transformed_images))
    sys.stdout.write("\n")


if __name__ == '__main__':
    path = '../../Datasets/chest_xray/small/224/train/PNEUMONIA/'
    print("Augmenting pneumonia images:")
    transform(8000, path)

    print("Augmenting normal images:")
    path = '../../Datasets/chest_xray/small/224/train/NORMAL/'
    transform(8000, path)