from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np


def multiply(img_name, value):
    im = np.array(Image.open(img_name))

    im_m = (im * value) % 255

    img = Image.fromarray(np.uint8(im_m))

    img.save('img/multiply.jpg')

def subtract(img_name, value):
    im = np.array(Image.open(img_name))

    im_m = (im - value) % 255

    img = Image.fromarray(np.uint8(im_m))

    img.save('img/subtract.jpg')

def add(img_name, value):
    im = np.array(Image.open(img_name))

    im_m = (im + value) % 255

    img = Image.fromarray(np.uint8(im_m))

    img.save('img/add.jpg')

def divide(img_name, value):
    im = np.array(Image.open(img_name))

    im_m = (im / value) % 255

    img = Image.fromarray(np.uint8(im_m))

    img.save('img/divide.jpg')
