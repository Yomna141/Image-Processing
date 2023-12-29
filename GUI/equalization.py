from PIL import Image, ImageOps

def equalize(img_name):
    # creating an image1 object
    im1 = Image.open(img_name)
    
    # applying equalize method
    im2 = ImageOps.equalize(im1, mask=None)

    # Save equalized image
    im2.save('img/equalized_image.jpg')

# Call the function to test
# equalize('your_image.jpg')  # Uncomment this line if you want to test the function
