from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

img = Image.open("misc_data\morpho-butterfly-1452254349NBZ.jpg")
img_array = np.array(img)
img_array = np.moveaxis(img_array, -1, 0)

def extract_patches(image_array, h_patch_size=64, w_patch_size=64):
    C, H, W = image_array.shape
    h_patches = H // h_patch_size
    w_patches = W // w_patch_size 

    image_array = image_array[:, :h_patches*h_patch_size, :w_patches*w_patch_size]
    patches = image_array.reshape(C, h_patches, h_patch_size, w_patches, w_patch_size)

    return image_array, patches, h_patches, w_patches

def get_reference_patch(patches, h_patches, w_patches, h_patch_index=None, w_patch_index=None):
    range_h_patch = np.arange(0, h_patches)
    range_w_patch = np.arange(0, w_patches)

    random_h_index = np.random.choice(range_h_patch)
    random_w_index = np.random.choice(range_w_patch)
    
    reference_patch = patches[:, random_h_index, :, random_w_index, :]
    return reference_patch

def similar_patches(image_array, patches, sim_across_channels):
    pass

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def calculate_similarity(patches, reference_patch):
    reference_patch = reference_patch[:, np.newaxis, :, np.newaxis, :]
    diffs = np.abs(patches - reference_patch)
    diffs = diffs.mean(axis=(2,4))
    similarity = sigmoid(-diffs)
    return similarity

image_array, patches, h_patches, w_patches  = extract_patches(img_array)
print(patches.shape)
reference_patch = get_reference_patch(patches, h_patches, w_patches)
similarity_matrix = (calculate_similarity(patches, reference_patch))
sim_across_channels = sigmoid(-(similarity_matrix).sum(axis=0))
topk_idx = np.argpartition(-sim_across_channels, 3-1, axis=1)[:, :3]
print(topk_idx)