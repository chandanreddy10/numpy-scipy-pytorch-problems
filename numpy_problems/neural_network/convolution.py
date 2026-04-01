import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

img = Image.open("..\image_patch_similarity\misc_data\drink-on-beach.jpg")
img_array = np.array(img)

kernel = np.array([[1, 0, -1, 2, 1],
                   [0, 1, 1, 2, 2],
                   [0,-1, -1, 1, 2],
                   [2, 1, 1, -1, 0],
                   [1, 2, 0, 1, 2]])
stride = 2 
padding = 0
output_array = np.array([[[]]])

