from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt


# Add salt and pepper noise
def salt_and_pepper(img_path, prob):
    # Load the image
    img = Image.open(img_path)

    # Convert the image to a numpy array
    img_array = np.array(img)
        # Create a copy of the image array
    noisy_array = np.copy(img_array)

    # Generate random noise
    rnd = np.random.rand(img_array.shape[0], img_array.shape[1])

    # Add salt noise
    noisy_array[rnd < prob/2] = 255

    # Add pepper noise
    noisy_array[rnd > 1 - prob/2] = 0

    img = Image.fromarray(np.uint8(noisy_array))

    img.save('img/noise_test.jpg')


# Add uniform noise
def uniform_noise(img_path, scale):
        # Load the image
    img = Image.open(img_path)

    # Convert the image to a numpy array
    img_array = np.array(img)
    # Create a copy of the image array
    noisy_array = np.copy(img_array)

    # Generate random noise
    noise = np.random.uniform(-scale, scale, img_array.shape)

    # Add noise to the image
    noisy_array = np.clip(noisy_array + noise, 0, 255)

    img = Image.fromarray(np.uint8(noisy_array))

    img.save('img/noise_test.jpg')

# Add Gaussian noise
def gaussian_noise(img_path, mean, std):
        # Load the image
    img = Image.open(img_path)

    # Convert the image to a numpy array
    img_array = np.array(img)
    # Create a copy of the image array
    noisy_array = np.copy(img_array)

    # Generate random noise
    noise = np.random.normal(mean, std, img_array.shape)

    # Add noise to the image
    noisy_array = np.clip(noisy_array + noise, 0, 255)

    img = Image.fromarray(np.uint8(noisy_array))

    img.save('img/noise_test.jpg')
# Add Rayleigh noise
def rayleigh_noise(img_path, scale):
        # Load the image
    img = Image.open(img_path)

    # Convert the image to a numpy array
    img_array = np.array(img)
    # Create a copy of the image array
    noisy_array = np.copy(img_array)

    # Generate random noise
    noise = np.random.rayleigh(scale, img_array.shape)

    # Add noise to the image
    noisy_array = np.clip(noisy_array + noise, 0, 255)

    img = Image.fromarray(np.uint8(noisy_array))

    img.save('img/noise_test.jpg')
# Add exponential noise
def exponential_noise(img_path, scale):
        # Load the image
    img = Image.open(img_path)

    # Convert the image to a numpy array
    img_array = np.array(img)
    # Create a copy of the image array
    noisy_array = np.copy(img_array)

    # Generate random noise
    noise = np.random.exponential(scale, img_array.shape)

    # Add noise to the image
    noisy_array = np.clip(noisy_array + noise, 0, 255)

    img = Image.fromarray(np.uint8(noisy_array))

    img.save('img/noise_test.jpg')

"""
# Load the image
img = Image.open('image processing\img\colored.jpg')

# Convert the image to a numpy array
img_array = np.array(img)


plt.figure(figsize=(15, 5))

plt.subplot(2, 6, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

noisy_array = exponential_noise(img_array, 50)

# Convert the noisy array back to an image
noisy_img = Image.fromarray(noisy_array.astype(np.uint8))

plt.subplot(2, 6, 2)
plt.imshow(noisy_img, cmap='gray')
plt.title('Original Image')

noisy_array = salt_and_pepper(img_array, 0.05)

# Convert the noisy array back to an image
noisy_img = Image.fromarray(noisy_array.astype(np.uint8))

plt.subplot(2, 6, 3)
plt.imshow(noisy_img, cmap='gray')
plt.title('Salt and Pepper noise')

noisy_array = uniform_noise(img_array, 50)

# Convert the noisy array back to an image
noisy_img = Image.fromarray(noisy_array.astype(np.uint8))

plt.subplot(2, 6, 4)
plt.imshow(noisy_img, cmap='gray')
plt.title('Uniform noise')

noisy_array = gaussian_noise(img_array, 0, 50)

# Convert the noisy array back to an image
noisy_img = Image.fromarray(noisy_array.astype(np.uint8))

plt.subplot(2, 6, 5)
plt.imshow(noisy_img, cmap='gray')
plt.title('Gaussian noise')

noisy_array = rayleigh_noise(img_array, 50)

# Convert the noisy array back to an image
noisy_img = Image.fromarray(noisy_array.astype(np.uint8))

plt.subplot(2, 6, 6)
plt.imshow(noisy_img, cmap='gray')
plt.title('Rayleigh noise')



plt.show()

"""