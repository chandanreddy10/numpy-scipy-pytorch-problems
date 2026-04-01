import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

#Expand the project to include multiple filters
img = Image.open("..\image_patch_similarity\misc_data\drink-on-beach.jpg").convert('L')  # 'L' = grayscale
img_array = np.array(img)


kernel = np.array([[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1]])

stride = 2
padding = 0

if padding > 0:
    img_array = np.pad(img_array, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

H_in, W_in = img_array.shape
K_h, K_w = kernel.shape

H_out = (H_in - K_h + 2*padding) // stride + 1
W_out = (W_in - K_w + 2*padding) // stride + 1

output_array = np.zeros((H_out, W_out))

for i in range(H_out):
    for j in range(W_out):
        h_start = i * stride
        h_end = h_start + K_h
        w_start = j * stride
        w_end = w_start + K_w
        
        patch = img_array[h_start:h_end, w_start:w_end]
        output_array[i, j] = np.sum(patch * kernel)


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_array, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Convolved Image")
plt.imshow(output_array, cmap='gray')
plt.axis('off')

plt.show()