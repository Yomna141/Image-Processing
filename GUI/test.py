import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class ImageHistogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Histogram")

        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Button to open an image
        open_button = tk.Button(root, text="Open Image", command=self.open_image)
        open_button.pack()

        # Button to show histogram
        histogram_button = tk.Button(root, text="Show Histogram", command=self.show_histogram)
        histogram_button.pack()

        self.image = None

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if file_path:
            self.image = Image.open(file_path)
            self.display_image()

    def display_image(self):
        # Resize the image to fit the window
        resized_image = self.image.resize((300, 300))
        tk_image = ImageTk.PhotoImage(resized_image)

        # Update the label with the new image
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image

    def show_histogram(self):
        if self.image:
            # Separate image into RGB channels
            r, g, b = self.image.split()

            # Calculate histograms for each channel
            histogram_values_r, bins_r = np.histogram(np.array(r).flatten(), bins=256, range=(0, 256), density=True)
            histogram_values_g, bins_g = np.histogram(np.array(g).flatten(), bins=256, range=(0, 256), density=True)
            histogram_values_b, bins_b = np.histogram(np.array(b).flatten(), bins=256, range=(0, 256), density=True)

            # Create a figure with subplots for each channel
            fig, axs = plt.subplots(3, 1, figsize=(6, 9))

            # Plot histograms for each channel
            axs[0].plot(bins_r[:-1], histogram_values_r, color='red')
            axs[0].set_title("Red Channel Histogram")
            axs[1].plot(bins_g[:-1], histogram_values_g, color='green')
            axs[1].set_title("Green Channel Histogram")
            axs[2].plot(bins_b[:-1], histogram_values_b, color='blue')
            axs[2].set_title("Blue Channel Histogram")

            for ax in axs:
                ax.set_xlabel("Pixel Value")
                ax.set_ylabel("Frequency")

            # Adjust layout
            plt.tight_layout()

            # Embed the matplotlib figure into the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=self.root)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack()

            # Display figure
            canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageHistogramApp(root)
    root.mainloop()
