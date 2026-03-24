from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.patches as ptch

img = Image.open("misc_data\drink-on-beach.jpg")
img_array = np.array(img)

# (C, H, W) -> (H, W, C)
img_array = np.moveaxis(img_array, -1, 0)

H_PATCH_SIZE = 16
W_PATCH_SIZE = 16


def extract_patches(image_array, h_patch_size=H_PATCH_SIZE, w_patch_size=W_PATCH_SIZE):
    """
    Extracts all the patches for given height and width patch size.
    Here, the focus is on reshape.
    """
    C, H, W = image_array.shape
    h_patches = H // h_patch_size
    w_patches = W // w_patch_size

    image_array = image_array[:, :h_patches * h_patch_size, :w_patches * w_patch_size]
    patches = image_array.reshape(C, h_patches, h_patch_size, w_patches, w_patch_size)

    return image_array, patches, h_patches, w_patches


def get_reference_patch(patches, h_patches, w_patches, h_patch_index=None, w_patch_index=None):
    """
    Returns a reference patch at given index else takes random patch
    """
    if not h_patch_index:
        range_h_patch = np.arange(0, h_patches)
        h_patch_index = np.random.choice(range_h_patch)

    if not w_patch_index:
        range_w_patch = np.arange(0, w_patches)
        w_patch_index = np.random.choice(range_w_patch)

    reference_patch = patches[:, h_patch_index, :, w_patch_index, :]
    return reference_patch, h_patch_index, w_patch_index


def return_similar_patches(image_array, sim_across_channels, topk=100,
                           h_patch_size=H_PATCH_SIZE, w_patch_size=W_PATCH_SIZE):
    """
    Returns the most similar patches
    """
    flat = sim_across_channels.flatten()

    indices = np.argpartition(flat, -topk)[-topk:]
    indices = indices[np.argsort(-flat[indices])]

    similar_indices = np.unravel_index(indices, sim_across_channels.shape)

    similar_indices_height = similar_indices[0] * h_patch_size
    similar_indices_width = similar_indices[1] * w_patch_size

    return (similar_indices_height, similar_indices_width)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_similarity(patches, reference_patch):
    reference_patch = reference_patch[:, np.newaxis, :, np.newaxis, :]

    dot = (patches * reference_patch).sum(axis=(0, 2, 4))

    patch_norm = np.sqrt((patches ** 2).sum(axis=(0, 2, 4)))
    ref_norm = np.sqrt((reference_patch ** 2).sum())

    sim = 1 - (dot / (patch_norm * ref_norm + 1e-8))

    return sim


def plot_similar_patches(image_array, similar_patches,
                         h_patch_index, w_patch_index,
                         h_patch_size=16, w_patch_size=16):

    img = image_array.transpose(1, 2, 0)

    h_patch = h_patch_index * h_patch_size
    w_patch = w_patch_index * w_patch_size

    fig, ax = plt.subplots()
    ax.matshow(img)

    for h, w in zip(similar_patches[0], similar_patches[1]):
        rect = ptch.Rectangle(
            (w, h),
            w_patch_size,
            h_patch_size,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

    rect = ptch.Rectangle(
        (w_patch, h_patch),
        w_patch_size,
        h_patch_size,
        linewidth=2,
        edgecolor='green',
        facecolor='none'
    )
    ax.add_patch(rect)

    plt.savefig("misc_data/similar_patches.jpg", dpi=300)
    plt.show()


def normalize_patches(p):
    mean = p.mean(axis=(2, 4), keepdims=True)
    std = p.std(axis=(2, 4), keepdims=True) + 1e-8
    return (p - mean) / std


def normalize_reference_patch(p):
    mean = p.mean(keepdims=True)
    std = p.std(keepdims=True) + 1e-8
    return (p - mean) / std


image_array, patches, h_patches, w_patches = extract_patches(img_array)

reference_patch, h_patch_index, w_patch_index = get_reference_patch(
    patches,
    h_patches,
    w_patches,
    h_patch_index=45,
    w_patch_index=30
)

patches = normalize_patches(patches)
reference_patch = normalize_reference_patch(reference_patch)

similarity_matrix = calculate_similarity(patches, reference_patch)
print(similarity_matrix)

similar_patches = return_similar_patches(
    image_array,
    similarity_matrix,
    topk=150,
    h_patch_size=16,
    w_patch_size=16
)

print(similar_patches)

plot_similar_patches(
    image_array,
    similar_patches,
    h_patch_index,
    w_patch_index
)