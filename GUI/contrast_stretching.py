from PIL import Image, ImageOps
import numpy as np

# Method to process the red band of the image
def normalizeRed(intensity):
    iI = intensity
    iO = (iI-86)*(((255-0)/(230-86))+0)
    return iO

# Method to process the green band of the image
def normalizeGreen(intensity):
    iI = intensity
    iO = (iI-90)*(((255-0)/(255-90))+0)
    return iO

# Method to process the blue band of the image
def normalizeBlue(intensity):
    iI = intensity
    iO = (iI-100)*(((255-0)/(210-100))+0)
    return iO

# Method to process the blue band of the image
def normalizeGray(intensity):
    iI = intensity
    iO = (iI-100)*(((255-0)/(210-100))+0)
    return iO

def stretching(img_name):
    # Create an image object
    imageObject = Image.open(img_name)

    # Split the red, green and blue bands from the Image
    multiBands = imageObject.split()

    # Apply point operations that does contrast stretching on each color band
    normalizedRedBand = multiBands[0].point(normalizeRed)
    normalizedGreenBand = multiBands[1].point(normalizeGreen)
    normalizedBlueBand = multiBands[2].point(normalizeBlue)

    # Create a new image from the contrast stretched rgb brands
    normalizedImage = Image.merge("RGB", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand))

    pil_img = Image.fromarray(np.uint8(normalizedImage))
    pil_img.save('img/after_contrast_stretching.png')

stretching("img/before_contrast_stretching.png")