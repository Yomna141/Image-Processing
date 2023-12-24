from PIL import Image
import numpy as np

def convert_to_binary(img_name, value):
    im = np.array(Image.open(img_name))
    im_m = im.copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            R, G, B = im[i, j]
            if (R + G + B > value):
                im_m[i, j][:] = 255
            else:
                im_m[i, j][:] = 0
    pil_img = Image.fromarray(np.uint8(im_m))
    pil_img.save('img/binary_test.jpg')

def convert_to_binary2(img_name, value):
    im = np.array(Image.open(img_name))
    im_m = im.copy()  # Copy the entire 2D array

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            pixel_value = im[i, j]

            # Check if any element in the pixel_value array is greater than the threshold
            if np.any(pixel_value > value):
                im_m[i, j] = 255
            else:
                im_m[i, j] = 0

    pil_img = Image.fromarray(np.uint8(im_m))
    pil_img.save('img/binary_test.jpg')


def convert_to_gray(img_name):
    im = np.array(Image.open(img_name))
    im_m = im[:, :, 0].copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            R, G, B = im[i, j]
            im_m[i][j]= ((R+G+B) / 3)

    pil_img = Image.fromarray(np.uint8(im_m))
    pil_img.save('img/gray_test.jpg')


def gray_to_binary(img_name, value):
    im = np.array(Image.open(img_name))

    im_m = im.copy()

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if (im[i][j] > value):
                im_m[i][j] = 255
            else:
                im_m[i][j] = 0

    pil_img = Image.fromarray(np.uint8(im_m))
    pil_img.save('img/binary_from_gray.jpg')
