import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageOps
import cv2
import numpy as np

# ... (previous code)

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        # Create a canvas with a vertical scrollbar
        self.canvas = tk.Canvas(root, bg="lightgray")
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nswe")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        # Center the frame in the main screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        # Set the background color of the frame
        self.scrollable_frame.configure(bg="lightgray")

        # Create a style object
        self.style = ttk.Style()

        # Configure the style for the buttons
        self.style.configure("TButton", background="lightgray", foreground="blue", padding=(10, 8), font=('Helvetica', 10, 'bold'))

        # Widgets
        self.upload_button = ttk.Button(self.scrollable_frame, text="Upload Image", command=self.upload_image, style="TButton")
        self.upload_button.grid(row=0, column=0, pady=10, padx=10, sticky="w")

        self.image_frame = ttk.Frame(self.scrollable_frame, relief="solid", borderwidth=2)
        self.image_frame.grid(row=1, column=0, rowspan=8, pady=10, padx=10, sticky="nswe")
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(self.image_frame, text="Uploaded Image")
        self.image_label.pack(fill="both", expand=True)

        # Create the processed image widget
        self.processed_image_widget = ProcessedImageWidget(self.scrollable_frame)
        self.processed_image_widget.grid(row=1, column=1, rowspan=8, pady=10, padx=10, sticky="nswe")
        self.processed_image_widget.columnconfigure(0, weight=1)
        self.processed_image_widget.rowconfigure(0, weight=1)

        # Button Grid
        button_grid = {
            "Grayscale": (2, 0),
            "Blur": (2, 1),
            "Invert Colors": (2, 2),
            "Multiply": (2, 3),
            "Subtract": (2, 4),
            "Add": (2, 5),
            "Divide": (2, 6),
            "Average Filter": (3, 0),
            "Min Filter": (3, 1),
            "Max Filter": (3, 2),
            "Median Filter": (3, 3),
            "Butterworth Low Pass": (3, 4),
            "Butterworth High Pass": (3, 5),
            "Sobel X Filter": (3, 6),
            "Sobel Y Filter": (4, 0),
            "Laplacian Filter": (4, 1),
            "Multiply": (4, 2),
            "Subtract": (4, 3),
            "Add": (4, 4),
            "Divide": (4, 5),
            "Show Light Gamma Image": (4, 6),
            "Show Dark Gamma Image": (5, 0),
        }

        for process, position in button_grid.items():
            button = ttk.Button(self.scrollable_frame, text=process, command=lambda p=process: self.process_image(p), style="TButton")
            button.grid(row=position[0], column=position[1], pady=5, padx=5, sticky="w")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
