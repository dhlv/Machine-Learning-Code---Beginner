from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

# Get the dimensions of the image
img = Image.open('images/my_img.jpg');
col = img.size[0]
row = img.size[1]

# load the image
img = plt.imread('images/my_img.jpg')
plt.subplot(221)
plt.imshow(img)

# Create two filters for horizontal and vertical detection
filter_for_horizontal = np.array([[-1, -1], [1, 1]])
filter_for_vertical = np.array([[-1, 1], [-1, 1]])

# Create matrix for filtered image
horizontal = np.zeros((row, col))

# multiple 2*2 matrix from the top-left cornor
for i in range(row):
    for j in range(col):
        temp = img[i:i + 2, j:j + 2, 1]
        temp = temp * filter_for_horizontal
        horizontal[i][j] = temp.sum()

# show the filtered image
plt.subplot(222)
plt.imshow(horizontal)

vertical = np.zeros((row, col))
for i in range(row):
    for j in range(col):
        temp = img[i:i + 2, j:j + 2, 1]
        temp = temp * filter_for_vertical
        vertical[i][j] = temp.sum()

plt.subplot(223)
plt.imshow(vertical)


# Define a function for getting the filter for gussian blur
# This method is learned from CSDN
# Source: https://blog.csdn.net/wl1710582732/article/details/78755037
def get_gussian_function_value(x, y):
    sigma = 4
    return (1 / (2 * math.pi * sigma ** 2)) * math.e ** (-(x ** 2 + y ** 2) / (2 * sigma ** 2))


radius = 10

# Get the matrix of my image
image_matrix = np.array(img)

# Create a 3*3 gussian blur filter
gussian_blur_filter = np.zeros((2 * radius + 1, 2 * radius + 1))
for i in range(2 * radius + 1):
    for j in range(2 * radius + 1):
        gussian_blur_filter[i, j] = get_gussian_function_value(i - radius, j - radius)
gussian_blur_filter /= gussian_blur_filter.sum()

print(gussian_blur_filter)
gussian_filtered_result = np.zeros((row, col))

# multiple gussian blur filter from the top-left cornor
for i in range(radius, row - radius):
    for j in range(radius, col - radius):
        temp = image_matrix[i - radius: i + radius + 1, j - radius: j + radius + 1, 1]
        temp = temp * gussian_blur_filter
        gussian_filtered_result[i][j] = temp.sum()

plt.subplot(224)
plt.imshow(gussian_filtered_result,
           cmap='gray')  # Why the color of the image is not the same as original even without 'gray'?

plt.tight_layout()
