# Importing Image and ImageFilter module from PIL package 
from PIL import Image, ImageFilter 
import numpy as np
import matplotlib.pyplot as plt


# creating an image object 
image_path = 'img/gray.jpg'
original_image = Image.open(image_path).convert('L') # Convert original to grayscale

# applying min filter 
img_min = original_image.filter(ImageFilter.MinFilter(size = 3)) 

# applying max filter 
img_max = original_image.filter(ImageFilter.MaxFilter(size = 3)) 

# applying median filter
img_median = original_image.filter(ImageFilter.MedianFilter(size = 3)) 