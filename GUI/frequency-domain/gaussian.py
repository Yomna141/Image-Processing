<<<<<<< HEAD:GUI/gaussian.py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter

def gaussian_low_pass_filter(size, radius):
    m, n = size
    center = (m // 2, n // 2)
    x, y = np.ogrid[:m, :n]
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    filter = np.exp(-distances**2 / (2 * (radius**2)))
    return filter

def gaussian_high_pass_filter(size, radius):
    low_pass_filter = gaussian_low_pass_filter(size, radius)
    return 1 - low_pass_filter

def apply_filter(image, filter):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    f_transform_filtered = f_transform_shifted * filter
    image_filtered = np.abs(ifft2(ifftshift(f_transform_filtered)))
    return image_filtered

def main():
    image_path = 'img/gray.jpg'
    original_image = np.array(Image.open(image_path).convert('L'))  # Convert original to grayscale

    original_image = original_image.astype(float) / 255.0

    filter_radius = 30

    high_pass_filter = gaussian_high_pass_filter(original_image.shape, filter_radius)
    sharpened_image_high = apply_filter(original_image, high_pass_filter)

    low_pass_filter = gaussian_low_pass_filter(original_image.shape, filter_radius)
    sharpened_image_low = apply_filter(original_image, low_pass_filter)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(sharpened_image_high, cmap='gray')
    plt.title('Gaussian High-Pass Sharpened')

    plt.subplot(1, 3, 3)
    plt.imshow(sharpened_image_low, cmap='gray')
    plt.title('Gaussian Low-Pass Sharpened')

    plt.show()

if __name__ == "__main__":
    main()
=======
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter

def gaussian_low_pass_filter(size, radius):
    m, n = size
    center = (m // 2, n // 2)
    x, y = np.ogrid[:m, :n]
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    filter = np.exp(-distances**2 / (2 * (radius**2)))
    return filter

def gaussian_high_pass_filter(size, radius):
    low_pass_filter = gaussian_low_pass_filter(size, radius)
    return 1 - low_pass_filter

def apply_filter(image, filter):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    f_transform_filtered = f_transform_shifted * filter
    image_filtered = np.abs(ifft2(ifftshift(f_transform_filtered)))
    return image_filtered

def main():
    image_path = 'img/gray.jpg'
    original_image = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale

    original_image = original_image.astype(float) / 255.0

    filter_radius = 30

    high_pass_filter = gaussian_high_pass_filter(original_image.shape, filter_radius)
    sharpened_image_high = apply_filter(original_image, high_pass_filter)

    low_pass_filter = gaussian_low_pass_filter(original_image.shape, filter_radius)
    sharpened_image_low = apply_filter(original_image, low_pass_filter)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(sharpened_image_high, cmap='gray')
    plt.title('Gaussian High-Pass Sharpened')

    plt.subplot(1, 3, 3)
    plt.imshow(sharpened_image_low, cmap='gray')
    plt.title('Gaussian Low-Pass Sharpened')

    plt.show()

if __name__ == "__main__":
    main()
>>>>>>> efd241b60112fa7f263803c64debf0404412c6bc:GUI/frequency-domain/gaussian.py
