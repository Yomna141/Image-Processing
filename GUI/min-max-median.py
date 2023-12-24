# Importing Image and ImageFilter module from PIL package 
from PIL import Image, ImageFilter 
import numpy as np
import matplotlib.pyplot as plt


# creating a image object 
image_path = 'img/gray.jpg'
original_image = Image.open(image_path).convert('L') # Convert to grayscale

# applying the min filter 
img_min = original_image.filter(ImageFilter.MinFilter(size = 3)) 

# applying the max filter 
img_max = original_image.filter(ImageFilter.MaxFilter(size = 3)) 

# applying the median filter
img_median = original_image.filter(ImageFilter.MedianFilter(size = 3)) 