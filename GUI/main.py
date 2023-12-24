import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageOps
import numpy as np
from average import averaging  # Import the averaging function
import matplotlib.pyplot as plt
from operations import multiply, subtract, add, divide
from gamma import gamma
from convertion import (
    convert_to_binary,
    convert_to_binary2,
    convert_to_gray,
    gray_to_binary,
)
from equalization import equalize  # Import the equalization function
from contrast_stretching import stretching
from scipy.ndimage import gaussian_filter
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from noise_types import (
    exponential_noise,
    gaussian_noise,
    rayleigh_noise, 
    salt_and_pepper, 
    uniform_noise
)
from histogram import show_histogram

class ProcessedImageWidget(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.image_label = ttk.Label(self, text="Processed Image")
        self.image_label.pack()

    def display_image(self, image):
        # Resize the image to a fixed size (e.g., 400x400)
        img = ImageTk.PhotoImage(image.resize((300, 300)))
        self.image_label.config(image=img)
        self.image_label.image = img


class BlurOptionsWidget(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.blur_label = ttk.Label(self, text="Blur Type:")
        self.blur_label.grid(row=0, column=0, pady=5, padx=10, sticky="w")

        self.blur_type = tk.StringVar()
        self.gaussian_radio = ttk.Radiobutton(self, text="Gaussian", variable=self.blur_type, value="gaussian")
        self.motion_radio = ttk.Radiobutton(self, text="Motion", variable=self.blur_type, value="motion")
        self.blur_type.set("gaussian")  # Default blur type
        self.gaussian_radio.grid(row=1, column=0, pady=5, padx=10, sticky="w")
        self.motion_radio.grid(row=2, column=0, pady=5, padx=10, sticky="w")





class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        # Set the initial size of the window
        self.root.geometry("1000x850")

        # Set the background color of the window
        self.root.configure(bg="lightgray")

        # Create a style object
        self.style = ttk.Style()

        # Configure the style for the buttons
        self.style.configure("TButton", background="blue", foreground="blue")

        # Widgets
        self.upload_button = ttk.Button(root, text="Upload Image", command=self.upload_image, style="TButton")
        self.upload_button.grid(row=0, column=0, pady=10, padx=10, sticky="w")

        self.image_frame = ttk.Frame(root, relief="solid", borderwidth=2)
        self.image_frame.grid(row=0, column=1, rowspan=5, pady=10, padx=10, sticky="nswe")
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        

        self.image_label = ttk.Label(self.image_frame, text="Uploaded Image")
        self.image_label.pack(fill="both", expand=True)

        self.process_options = ["Grayscale", "Blur", "Invert Colors"]
        self.process_buttons = {}

        for i, process in enumerate(self.process_options):
            button = ttk.Button(root, text=process, command=lambda p=process: self.process_image(p), style="TButton")
            button.grid(row=i + 1, column=0, pady=10, padx=10, sticky="w")
            self.process_buttons[process] = {'row': i + 1, 'column': 0}

        # Create the BlurOptionsWidget
        self.blur_options_widget = BlurOptionsWidget(root)
        self.blur_options_widget.grid(row=len(self.process_options) + 1, column=0, pady=10, padx=10, sticky="w")


        self.process_button = ttk.Button(root, text="Process Image", command=self.process_image, style="TButton")
        self.process_button.grid(row=len(self.process_options) + 2, column=0, pady=20, padx=10, sticky="w")

        # Create the processed image widget
        self.processed_image_widget = ProcessedImageWidget(root)
        self.processed_image_widget.grid(row=0, column=2, rowspan=5, pady=10, padx=10, sticky="nswe")
        self.processed_image_widget.columnconfigure(0, weight=1)
        self.processed_image_widget.rowconfigure(0, weight=1)

        # Add a new button for the averaging filter
        self.average_button = ttk.Button(root, text="Average Filter", command=self.apply_average_filter, style="TButton")
        self.average_button.grid(row=len(self.process_options) + 3, column=0, pady=10, padx=10, sticky="w")

        # Move the new buttons next to the "Average Filter" button
        self.min_button = ttk.Button(root, text="Min Filter", command=self.apply_min_filter, style="TButton")
        self.min_button.grid(row=len(self.process_options) + 3, column=1, pady=10, padx=10, sticky="w")

        self.max_button = ttk.Button(root, text="Max Filter", command=self.apply_max_filter, style="TButton")
        self.max_button.grid(row=len(self.process_options) + 3, column=2, pady=10, padx=10, sticky="w")

        self.median_button = ttk.Button(root, text="Median Filter", command=self.apply_median_filter, style="TButton")
        self.median_button.grid(row=len(self.process_options) + 3, column=3, pady=10, padx=10, sticky="w")
        
        self.butterworth_low_pass_button = ttk.Button(
            root,
            text="Butterworth Low Pass",
            command=self.apply_butterworth_low_pass,
            style="TButton",
        )
        self.butterworth_low_pass_button.grid(
            row=len(self.process_options) + 4, column=0, pady=10, padx=10, sticky="w"
        )

        self.butterworth_high_pass_button = ttk.Button(
            root,
            text="Butterworth High Pass",
            command=self.apply_butterworth_high_pass,
            style="TButton",
        )
        self.butterworth_high_pass_button.grid(
            row=len(self.process_options) + 4, column=1, pady=10, padx=10, sticky="w"
        )
                
        self.sobel_x_button = ttk.Button(
            root,
            text="Sobel X Filter",
            command=self.apply_sobel_x_filter,
            style="TButton",
        )
        self.sobel_x_button.grid(
            row=len(self.process_options) + 5, column=0, pady=10, padx=10, sticky="w"
        )

        self.sobel_y_button = ttk.Button(
            root,
            text="Sobel Y Filter",
            command=self.apply_sobel_y_filter,
            style="TButton",
        )
        self.sobel_y_button.grid(
            row=len(self.process_options) + 5, column=1, pady=10, padx=10, sticky="w"
        )

        self.laplacian_button = ttk.Button(
            root,
            text="Laplacian Filter",
            command=self.apply_laplacian_filter,
            style="TButton",
        )
        self.laplacian_button.grid(
            row=len(self.process_options) + 5, column=2, pady=10, padx=10, sticky="w"
        )
        
        self.multiply_button = ttk.Button(root, text="Multiply", command=lambda: self.process_image("Multiply"), style="TButton")
        self.multiply_button.grid(row=len(self.process_options) + 6, column=0, pady=10, padx=10, sticky="w")

        self.subtract_button = ttk.Button(root, text="Subtract", command=lambda: self.process_image("Subtract"), style="TButton")
        self.subtract_button.grid(row=len(self.process_options) + 6, column=1, pady=10, padx=10, sticky="w")

        self.add_button = ttk.Button(root, text="Add", command=lambda: self.process_image("Add"), style="TButton")
        self.add_button.grid(row=len(self.process_options) + 6, column=2, pady=10, padx=10, sticky="w")

        self.divide_button = ttk.Button(root, text="Divide", command=lambda: self.process_image("Divide"), style="TButton")
        self.divide_button.grid(row=len(self.process_options) + 6, column=3, pady=10, padx=10, sticky="w")
        
        self.light_gamma_button = ttk.Button(
            root, text="Show Light Gamma Image", command=self.display_light_gamma, style="TButton"
        )
        self.light_gamma_button.grid(row=len(self.process_options) + 7, column=0, pady=10, padx=10, sticky="w")

        self.dark_gamma_button = ttk.Button(
            root, text="Show Dark Gamma Image", command=self.display_dark_gamma, style="TButton"
        )
        self.dark_gamma_button.grid(row=len(self.process_options) + 7, column=1, pady=10, padx=10, sticky="w")
        
        self.binary_convert_button = ttk.Button(
            root, text="Convert to Binary", command=self.convert_to_binary, style="TButton"
        )
        self.binary_convert_button.grid(row=len(self.process_options) + 8, column=0, pady=10, padx=10, sticky="w")

        self.binary_convert2_button = ttk.Button(
            root, text="Convert to Binary (Method 2)", command=self.convert_to_binary2, style="TButton"
        )
        self.binary_convert2_button.grid(row=len(self.process_options) + 8, column=1, pady=10, padx=10, sticky="w")

        self.gray_convert_button = ttk.Button(
            root, text="Convert to Gray", command=self.convert_to_gray, style="TButton"
        )
        self.gray_convert_button.grid(row=len(self.process_options) + 8, column=2, pady=10, padx=10, sticky="w")

        self.gray_to_binary_button = ttk.Button(
            root, text="Gray to Binary", command=self.gray_to_binary, style="TButton"
        )
        self.gray_to_binary_button.grid(row=len(self.process_options) + 8, column=3, pady=10, padx=10, sticky="w")
        
        self.equalize_button = ttk.Button(
            root, text="Equalize", command=self.apply_equalization, style="TButton"
        )
        self.equalize_button.grid(row=len(self.process_options) + 7, column=2, pady=10, padx=10, sticky="w")
        
        self.stretch_button = ttk.Button(
            root, text="Stretch Image", command=self.apply_contrast_stretching, style="TButton"
        )
        self.stretch_button.grid(row=len(self.process_options) + 9, column=3, pady=10, padx=10, sticky="w")
        
        self.display_red_button = ttk.Button(
            root, text="Display Red Channel", command=self.display_red_channel, style="TButton"
        )
        self.display_red_button.grid(row=len(self.process_options) + 9, column=0, pady=10, padx=10, sticky="w")

        self.display_green_button = ttk.Button(
            root, text="Display Green Channel", command=self.display_green_channel, style="TButton"
        )
        self.display_green_button.grid(row=len(self.process_options) + 9, column=1, pady=10, padx=10, sticky="w")

        self.display_blue_button = ttk.Button(
            root, text="Display Blue Channel", command=self.display_blue_channel, style="TButton"
        )
        self.display_blue_button.grid(row=len(self.process_options) + 9, column=2, pady=10, padx=10, sticky="w")
        
        self.gaussian_low_pass_button = ttk.Button(
            root,
            text="Gaussian Low Pass",
            command=self.apply_gaussian_low_pass,
            style="TButton",
        )
        self.gaussian_low_pass_button.grid(
            row=len(self.process_options) + 10, column=0, pady=10, padx=10, sticky="w"
        )

        self.gaussian_high_pass_button = ttk.Button(
            root,
            text="Gaussian High Pass",
            command=self.apply_gaussian_high_pass,
            style="TButton",
        )
        self.gaussian_high_pass_button.grid(
            row=len(self.process_options) + 10, column=1, pady=10, padx=10, sticky="w"
        )
        
        self.ideal_low_pass_button = ttk.Button(
            root,
            text="Ideal Low Pass",
            command=self.apply_ideal_low_pass,
            style="TButton",
        )
        self.ideal_low_pass_button.grid(
            row=len(self.process_options) + 10, column=2, pady=10, padx=10, sticky="w"
        )

        self.ideal_high_pass_button = ttk.Button(
            root,
            text="Ideal High Pass",
            command=self.apply_ideal_high_pass,
            style="TButton",
        )
        self.ideal_high_pass_button.grid(
            row=len(self.process_options) + 10, column=3, pady=10, padx=10, sticky="w"
        )

        self.salt_pepper_button = ttk.Button(
            root, 
            text="Salt and Pepper", 
            command=self.apply_salt_and_pepper_noise, 
            style="TButton",

        )

        self.salt_pepper_button.grid(            
            row=len(self.process_options) + 11, column=0, pady=10, padx=10, sticky="w"
            
        )
        
        self.uniform_noise_button = ttk.Button(
            root, 
            text="Uniform", 
            command=self.apply_uniform_noise, 
            style="TButton",
        )

        self.uniform_noise_button.grid(            
            row=len(self.process_options) + 11, column=1, pady=10, padx=10, sticky="w"   
        )

        self.gaussian_noise_button = ttk.Button(
            root, 
            text="Gaussian", 
            command=self.apply_gaussian_noise, 
            style="TButton",

        )
        self.gaussian_noise_button.grid(            
            row=len(self.process_options) + 11, column=2, pady=10, padx=10, sticky="w"
            
        )
        self.rayleigh_noise_button = ttk.Button(
            root, text="Rayleigh", 
            command=self.apply_rayleigh_noise, 
            style="TButton",

        )
        self.rayleigh_noise_button.grid(            
            row=len(self.process_options) + 11, column=3, pady=10, padx=10, sticky="w"
            
        )
        self.exponential_noise_button = ttk.Button(
            root, 
            text="Exponential", 
            command=self.apply_exponential_noise, 
            style="TButton",

        )
        self.exponential_noise_button.grid(            
            row=len(self.process_options) + 12, column=0, pady=10, padx=10, sticky="w"
            
        )
        self.histogram = ttk.Button(
            root, 
            text="Histogram", 
            command=self.apply_show_histogram, 
            style="TButton"

        )

        self.histogram.grid(            
            row=len(self.process_options) + 12, column=1, pady=10, padx=10, sticky="w"
        )

     
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*")])
        if file_path:
            self.image_path = file_path
            self.original_image = Image.open(self.image_path)
            self.display_image(self.original_image) 

    def display_image(self, image):
        img = ImageTk.PhotoImage(image.resize((300, 300)))  # Fixed size for uploaded image
        self.image_label.config(image=img)
        self.image_label.image = img

    def process_image(self, process=None):
        if self.original_image:
            selected_process = process or self.process_variable.get()

            if selected_process == "Grayscale":
                processed_image = self.original_image.convert("L")
            elif selected_process == "Blur":
                blur_type = self.blur_options_widget.blur_type.get()
                if blur_type == "gaussian":
                    processed_image = self.original_image.filter(ImageFilter.GaussianBlur(radius=5))
                elif blur_type == "motion":
                    processed_image = self.original_image.filter(ImageFilter.MotionBlur(angle=45, radius=10))
                else:
                    messagebox.showwarning("Warning", "Please select a valid blur type.")
                    return

            elif selected_process == "Noise":
                processed_image = self.original_image.convert("L")

            elif selected_process == "Invert Colors":
                processed_image = ImageOps.invert(self.original_image.convert("RGB"))
            elif selected_process == "Multiply":
                multiply(self.image_path, 1.5)
                processed_image = Image.open('img/multiply.jpg')
            elif selected_process == "Subtract":
                subtract(self.image_path, 20)
                processed_image = Image.open('img/subtract.jpg')
            elif selected_process == "Add":
                add(self.image_path, 30)
                processed_image = Image.open('img/add.jpg')
            elif selected_process == "Divide":
                divide(self.image_path, 2)
                processed_image = Image.open('img/divide.jpg')

            else:
                messagebox.showwarning("Warning", "Please select a valid process.")
                return

            self.processed_image_widget.display_image(processed_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def apply_average_filter(self):
        if hasattr(self, 'original_image'):
            # Use the averaging function from the average module
            averaged_image = averaging(self.image_path)

            # Convert the data type to np.uint8
            averaged_image = averaged_image.astype(np.uint8)

            # Display the processed image
            self.processed_image_widget.display_image(Image.fromarray(averaged_image))
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def apply_min_filter(self):
        if hasattr(self, 'original_image'):
            # Apply the min filter
            min_filtered_image = self.original_image.filter(ImageFilter.MinFilter(size=3))
            self.processed_image_widget.display_image(min_filtered_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def apply_max_filter(self):
        if hasattr(self, 'original_image'):
            # Apply the max filter
            max_filtered_image = self.original_image.filter(ImageFilter.MaxFilter(size=3))
            self.processed_image_widget.display_image(max_filtered_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def apply_median_filter(self):
        if hasattr(self, 'original_image'):
            # Apply the median filter
            median_filtered_image = self.original_image.filter(ImageFilter.MedianFilter(size=3))
            self.processed_image_widget.display_image(median_filtered_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")
            
    def apply_butterworth_low_pass(self):
        if self.original_image.mode != 'L':
                messagebox.showwarning("Warning", "Please upload a grayscale image.")
                return
        if hasattr(self, 'original_image'):
            # Convert the image to a numpy array
            image_array = np.array(self.original_image)

            # Apply Fourier Transform
            f_transform = fft2(image_array)
            f_transform_shifted = fftshift(f_transform)

            # Create a Butterworth low-pass filter
            rows, cols = image_array.shape
            center_row, center_col = rows // 2, cols // 2
            cutoff_frequency = 30  # You can adjust this cutoff frequency
            butterworth_order = 2  # You can adjust the filter order

            # Create the Butterworth low-pass filter mask
            x = np.arange(cols)
            y = np.arange(rows)
            x, y = np.meshgrid(x, y)
            mask = 1 / (1 + ((x - center_col) / cutoff_frequency)**(2 * butterworth_order) +
                        ((y - center_row) / cutoff_frequency)**(2 * butterworth_order))

            # Apply the Butterworth low-pass filter
            f_transform_shifted_filtered = f_transform_shifted * mask

            # Apply Inverse Fourier Transform
            image_array_filtered = np.abs(ifft2(ifftshift(f_transform_shifted_filtered)))

            # Convert the result back to PIL Image
            butterworth_low_pass_image = Image.fromarray(image_array_filtered.astype(np.uint8))

            # Display the processed image
            self.processed_image_widget.display_image(butterworth_low_pass_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def apply_butterworth_high_pass(self):
        if hasattr(self, 'original_image'):
            if self.original_image.mode != 'L':
                messagebox.showwarning("Warning", "Please upload a grayscale image.")
                return
            # Convert the image to a numpy array
            image_array = np.array(self.original_image)

            # Apply Fourier Transform
            f_transform = fft2(image_array)
            f_transform_shifted = fftshift(f_transform)

            # Create a Butterworth high-pass filter
            rows, cols = image_array.shape
            center_row, center_col = rows // 2, cols // 2
            cutoff_frequency = 30  # You can adjust this cutoff frequency
            butterworth_order = 2  # You can adjust the filter order

            # Create the Butterworth high-pass filter mask
            x = np.arange(cols)
            y = np.arange(rows)
            x, y = np.meshgrid(x, y)
            mask = 1 / (1 + ((x - center_col) / cutoff_frequency)**(2 * butterworth_order) +
                        ((y - center_row) / cutoff_frequency)**(2 * butterworth_order))
            mask = 1 - mask

            # Apply the Butterworth high-pass filter
            f_transform_shifted_filtered = f_transform_shifted * mask

            # Apply Inverse Fourier Transform
            image_array_filtered = np.abs(ifft2(ifftshift(f_transform_shifted_filtered)))

            # Convert the result back to PIL Image
            butterworth_high_pass_image = Image.fromarray(image_array_filtered.astype(np.uint8))

            # Display the processed image
            self.processed_image_widget.display_image(butterworth_high_pass_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")
            
    def apply_sobel_x_filter(self):
        if hasattr(self, 'original_image'):
            # Use the imported function
            filtered_image = self.apply_sobel_cv2(self.original_image, "x")

            # Display the processed image
            self.processed_image_widget.display_image(filtered_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def apply_sobel_y_filter(self):
        if hasattr(self, 'original_image'):
            # Use the imported function
            filtered_image = self.apply_sobel_cv2(self.original_image, "y")

            # Display the processed image
            self.processed_image_widget.display_image(filtered_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def apply_laplacian_filter(self):
        if hasattr(self, 'original_image'):
            # Use the imported function
            filtered_image = self.apply_laplacian_cv2(self.original_image)

            # Display the processed image
            self.processed_image_widget.display_image(filtered_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def apply_sobel_cv2(self, image, orientation):
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Remove noise using GaussianBlur
        img = cv2.GaussianBlur(gray, (3, 3), 0)

        # Convolute with the Sobel kernels
        if orientation == "x":
            filtered_image = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x
        elif orientation == "y":
            filtered_image = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y

        # Optional: Adjust the result to the 0-255 range
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)

        return Image.fromarray(filtered_image)

    def apply_laplacian_cv2(self, image):
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Remove noise using GaussianBlur
        img = cv2.GaussianBlur(gray, (3, 3), 0)

        # Convolute with the Laplacian kernel
        laplacian = cv2.Laplacian(img, cv2.CV_64F)

        # Optional: Adjust the result to the 0-255 range
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

        return Image.fromarray(laplacian)
    
    def display_light_gamma(self):
        if hasattr(self, 'original_image'):
            # Apply the gamma function to create the light image
            gamma(self.image_path)

            # Display the light image
            light_gamma_image = Image.open('img/gamma_lighter1.jpg')
            self.processed_image_widget.display_image(light_gamma_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def display_dark_gamma(self):
        if hasattr(self, 'original_image'):
            # Apply the gamma function to create the dark image
            gamma(self.image_path)

            # Display the dark image
            dark_gamma_image = Image.open('img/gamma_darker1.jpg')
            self.processed_image_widget.display_image(dark_gamma_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")
            
    def convert_to_binary(self):
        if hasattr(self, 'original_image'):
        # Check if the image is colored
            if len(np.array(self.original_image).shape) == 3:
            # Replace 'value' with the desired threshold value
                convert_to_binary(self.image_path, value=128)

            # Display the converted image
                converted_image = Image.open('img/binary_test.jpg')
                self.processed_image_widget.display_image(converted_image)
            else:
                messagebox.showwarning("Warning", "Please upload a colored image.")
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")


    def convert_to_binary2(self):
        if hasattr(self, 'original_image'):
            # Replace 'value' with the desired threshold value
            convert_to_binary2(self.image_path, value=128)

            # Display the converted image
            converted_image = Image.open('img/binary_test.jpg')
            self.processed_image_widget.display_image(converted_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def convert_to_gray(self):
        if hasattr(self, 'original_image'):
        # Check if the image is already grayscale
            if self.original_image.mode == 'L':
                messagebox.showinfo("Info", "The uploaded image is already in grayscale.")
                return

        # Convert the image to grayscale
            convert_to_gray(self.image_path)

        # Display the converted image
            converted_image = Image.open('img/gray_test.jpg')
            self.processed_image_widget.display_image(converted_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def gray_to_binary(self):
        if hasattr(self, 'original_image'):
        # Check if the image is grayscale
            if self.original_image.mode != 'L':
                messagebox.showwarning("Warning", "Please upload a grayscale image.")
                return

        # Replace 'value' with the desired threshold value
            gray_to_binary(self.image_path, value=128)

        # Display the converted image
            converted_image = Image.open('img/binary_from_gray.jpg')
            self.processed_image_widget.display_image(converted_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")
    
    def apply_equalization(self):
        if hasattr(self, 'original_image'):
            # Apply the equalization function
            equalize(self.image_path)

            # Display the equalized image
            equalized_image = Image.open('img/equalized_image.jpg')
            self.processed_image_widget.display_image(equalized_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")
            
    def apply_contrast_stretching(self):
        if hasattr(self, 'original_image'):
            if self.original_image.mode == 'L':
                messagebox.showinfo("Info", "Upload RGB image.")
                return
            # Apply the contrast-stretching function
            stretching(self.image_path)

            # Display the stretched image
            stretched_image = Image.open('img/after_contrast_stretching.png')
            self.processed_image_widget.display_image(stretched_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")
            
    def display_red_channel(self):
        if hasattr(self, 'original_image'):
            if self.original_image.mode == 'L':
                messagebox.showinfo("Info", "Upload RGB image.")
                return
            # Display the red channel
            red_channel = self.original_image.split()[0]
            self.processed_image_widget.display_image(red_channel)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def display_green_channel(self):
        if hasattr(self, 'original_image'):
            if self.original_image.mode == 'L':
                messagebox.showinfo("Info", "Upload RGB image.")
                return
            # Display the green channel
            green_channel = self.original_image.split()[1]
            self.processed_image_widget.display_image(green_channel)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def display_blue_channel(self):
        if hasattr(self, 'original_image'):
            if self.original_image.mode == 'L':
                messagebox.showinfo("Info", "Upload RGB image.")
                return
            # Display the blue channel
            blue_channel = self.original_image.split()[2]
            self.processed_image_widget.display_image(blue_channel)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")
            
    def apply_gaussian_low_pass(self):
        if self.original_image.mode != 'L':
                messagebox.showwarning("Warning", "Please upload a grayscale image.")
                return
            
        if hasattr(self, 'original_image'):
            # Convert the image to a numpy array
            image_array = np.array(self.original_image)

            # Set the standard deviation for the Gaussian filter
            sigma = 2

            # Apply the Gaussian filter to the image array
            low_pass_image_array = gaussian_filter(image_array, sigma=sigma)

            # Convert the result back to PIL Image
            low_pass_image = Image.fromarray(low_pass_image_array.astype(np.uint8))

            # Display the processed image
            self.processed_image_widget.display_image(low_pass_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def apply_gaussian_high_pass(self):
        if hasattr(self, 'original_image'):
            if self.original_image.mode != 'L':
                messagebox.showwarning("Warning", "Please upload a grayscale image.")
                return
            
            # Convert the image to a numpy array
            image_array = np.array(self.original_image)

            # Set the standard deviation for the Gaussian filter
            sigma = 2

            # Apply the Gaussian filter to the image array
            low_pass_image_array = gaussian_filter(image_array, sigma=sigma)

            # Calculate the high-pass image by subtracting the low-pass image from the original image
            high_pass_image_array = image_array - low_pass_image_array

            # Convert the result back to PIL Image
            high_pass_image = Image.fromarray(high_pass_image_array.astype(np.uint8))

            # Display the processed image
            self.processed_image_widget.display_image(high_pass_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")
            
    def apply_ideal_high_pass(self):
        if hasattr(self, 'original_image'):
            if self.original_image.mode != 'L':
                messagebox.showwarning("Warning", "Please upload a grayscale image.")
                return
            
            # Convert the image to a numpy array
            image_array = np.array(self.original_image)

            # Apply Fourier Transform
            f_transform = fft2(image_array)
            f_transform_shifted = fftshift(f_transform)

            # Create an ideal low-pass filter
            rows, cols = image_array.shape
            center_row, center_col = rows // 2, cols // 2
            radius = 30  # You can adjust this cutoff radius
            mask = np.ones((rows, cols), dtype=np.uint8)
            mask[(center_row - radius):(center_row + radius), (center_col - radius):(center_col + radius)] = 0

            # Apply the ideal low-pass filter
            f_transform_shifted_filtered = f_transform_shifted * mask

            # Apply Inverse Fourier Transform
            image_array_filtered = np.abs(ifft2(ifftshift(f_transform_shifted_filtered)))

            # Convert the result back to PIL Image
            ideal_low_pass_image = Image.fromarray(image_array_filtered.astype(np.uint8))

            # Display the processed image
            self.processed_image_widget.display_image(ideal_low_pass_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def apply_ideal_low_pass(self):
        if hasattr(self, 'original_image'):
            if self.original_image.mode != 'L':
                messagebox.showwarning("Warning", "Please upload a grayscale image.")
                return
            
            # Convert the image to a numpy array
            image_array = np.array(self.original_image)

            # Apply Fourier Transform
            f_transform = fft2(image_array)
            f_transform_shifted = fftshift(f_transform)

            # Create an ideal high-pass filter
            rows, cols = image_array.shape
            center_row, center_col = rows // 2, cols // 2
            radius = 30  # You can adjust this cutoff radius
            mask = np.zeros((rows, cols), dtype=np.uint8)
            mask[(center_row - radius):(center_row + radius), (center_col - radius):(center_col + radius)] = 1

            # Apply the ideal high-pass filter
            f_transform_shifted_filtered = f_transform_shifted * mask

            # Apply Inverse Fourier Transform
            image_array_filtered = np.abs(ifft2(ifftshift(f_transform_shifted_filtered)))

            # Convert the result back to PIL Image
            ideal_high_pass_image = Image.fromarray(image_array_filtered.astype(np.uint8))

            # Display the processed image
            self.processed_image_widget.display_image(ideal_high_pass_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")


    def apply_exponential_noise(self):
        if hasattr(self, 'original_image'):
            # Replace 'value' with the desired threshold value
            exponential_noise(self.image_path, 50)

            # Display the converted image
            converted_image = Image.open('img/noise_test.jpg')
            self.processed_image_widget.display_image(converted_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")
    def apply_salt_and_pepper_noise(self):
        if hasattr(self, 'original_image'):
            # Replace 'value' with the desired threshold value
            salt_and_pepper(self.image_path, 0.05)

            # Display the converted image
            converted_image = Image.open('img/noise_test.jpg')
            self.processed_image_widget.display_image(converted_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

        
    def apply_gaussian_noise(self):
        if hasattr(self, 'original_image'):
            # Replace 'value' with the desired threshold value
            gaussian_noise(self.image_path, 0, 50)

            # Display the converted image
            converted_image = Image.open('img/noise_test.jpg')
            self.processed_image_widget.display_image(converted_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")


    def apply_uniform_noise(self):
        if hasattr(self, 'original_image'):
            # Replace 'value' with the desired threshold value
            uniform_noise(self.image_path, 50)

            # Display the converted image
            converted_image = Image.open('img/noise_test.jpg')
            self.processed_image_widget.display_image(converted_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def apply_rayleigh_noise(self):
        if hasattr(self, 'original_image'):
            # Replace 'value' with the desired threshold value
            rayleigh_noise(self.image_path, 50)

            # Display the converted image
            converted_image = Image.open('img/noise_test.jpg')
            self.processed_image_widget.display_image(converted_image)
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

    def apply_show_histogram(self):
        if hasattr(self, 'original_image'):
             show_histogram(self.image_path)

        else:
            messagebox.showwarning("Warning", "Please upload an image first.")


        
    
  
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()