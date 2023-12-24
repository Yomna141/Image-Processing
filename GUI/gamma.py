from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

def gamma(img_name):
    im = np.array(Image.open(img_name))
    im_m = 255.0 * (im / 255.0) ** (1/22)
    im_m_2 = 255.0 * (im / 255.0) ** 2.2
    img1 = Image.fromarray(np.uint8(im_m))
    img1.save('img/gamma_lighter1.jpg')
    img2 = Image.fromarray(np.uint8(im_m_2))
    img2.save('img/gamma_darker1.jpg')


"""""""""
im = np.array(Image.open('test.jpg'))
img = Image.fromarray(im)
img.show()
"""""""""