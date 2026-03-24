from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

img = Image.open("misc_data\morpho-butterfly-1452254349NBZ.jpg")
img_array = np.array(img)
img_array = np.moveaxis(img_array, -1, 0)
print(img_array.shape)
def extract_patches(patch_size=128):
    pass

def similar_patches(k=3):
    pass

def calculate_distance(reference_patch, all_patches):
    pass

def get_reference_patch(image_array, h_patch_size=64, w_patch_size=64, h_patch_index=None, w_patch_index=None):
    C, H, W = image_array.shape
    h_patches = H // h_patch_size
    w_patches = W // w_patch_size 

    image_array = image_array[:, :h_patches*h_patch_size, :w_patches*w_patch_size]
    reference_patches = image_array.reshape(C, h_patches, h_patch_size, w_patches, w_patch_size)

    range_h_patch = np.arange(0, h_patches)
    range_w_patch = np.arange(0, w_patches)

    random_h_index = np.random.choice(range_h_patch)
    random_w_index = np.random.choice(range_w_patch)

    return reference_patches[:, random_h_index,:,random_w_index,:].shape

print(get_reference_patch(img_array))
    