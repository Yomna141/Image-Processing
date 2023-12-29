import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image (assuming a binary image for simplicity)
image = cv2.imread('binary.jpg', cv2.IMREAD_GRAYSCALE)

# Define a structuring element (kernel) - a 3x3 square in this example
kernel = np.ones((3, 3), np.uint8)

# Apply erosion to image
eroded_image = cv2.erode(image, kernel, iterations=1)

# Display the original and eroded images side by side
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(eroded_image, cmap='gray'), plt.title('Eroded')
plt.show()
