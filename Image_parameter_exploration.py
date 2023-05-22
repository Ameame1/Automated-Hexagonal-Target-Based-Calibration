'''
    This Code is for CITS4402: Computer Vision Lab05
    Name: Yuliang Zhang
    Student No.: 22828187
    Lab: Lab 05 Week 06

    Code Running description:
    - This code is transfered from the Lab04

    - The Colour based segmentation algorithm is cv2.inRange, the value of different components in HSV are as follows:
        - H: [lower_threshold, upper_threshold], according to our customed value, note that if the upper_threshold is
             smaller than lower_threshold, the upper_threshold will be adjusted to as the same as lower_threshold.
        - S: [0, 255], we will not adjust the value of S according to the submission requirement.
        - V: [0, 255], we will not adjust the value of V according to the submission requirement.

    - Best values for segmentation of Iris and Pepper images:
        - Iris:
            - lower_threshold [0,3]
            - upper_threshold [20, 80]

        - Peppers for green color:
            - lower_threshold [21,25]
            - upper_threshold [50, 100]

        - The Default value of upper_threshold has been adjusted to 75,so we do not need to adjust the upper_threshold

'''


import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
from target_localization import *

class ImageGUI:

    def __init__(self, master):
        self.master = master
        self.master.title("Colour based image segmentation")

        # [1] Fix the window size
        self.master.geometry("1280x1000")

        # Add the title at the top of the window
        self.title_text = tk.Label(self.master,
                                   text="Colour Based Image Segmentation",
                                   width=700,
                                   font=("Arial", 14),
                                   fg="yellow",
                                   background="#A52A2A")
        self.title_text.pack(side=tk.TOP, padx=5, pady=5)

        """
        Set the image display area:
            - Create a display frame
            - Create a canvas and set it to the left grid of the display frame (original image)
            - Create a canvas and set it to the right grid of the display frame (processed image)
        """
        # [2] Create the Image Displaying Area
        self.display_frame = tk.Frame(self.master, width=1000, height=500)
        self.display_frame.pack()
        self.display_frame.grid_rowconfigure(0, weight=1)
        self.display_frame.grid_columnconfigure(0, weight=1)

        # [2.1] Left area is for original image
        self.orig_img_canvas = tk.Canvas(master=self.display_frame, width=500, height=500)
        self.orig_img_canvas.grid(row=0, column=0, padx=20, pady=20)

        # [2.2] Right areas is for processed image
        self.processed_img_canvas = tk.Canvas(master=self.display_frame, width=500, height=500)
        self.processed_img_canvas.grid(row=0, column=1, padx=20, pady=20)

        """
        Set the first operating buttons: Load Image, Target Locate
        """
        # [3] Create the second area frame for the operating buttons
        self.button_area1 = tk.Frame(self.master)
        self.button_area1.pack(expand=True, padx=10, pady=10)
        self.button_area1.grid_rowconfigure(0, weight=1)
        self.button_area1.grid_columnconfigure(0, weight=1)

        # Configure columns to have the same width
        # self.button_area1.columnconfigure(0, uniform="column")
        self.button_area1.columnconfigure(1, uniform="column")

        # [3.2] Create a "Load Image" button
        self.load_border = tk.Frame(self.button_area1, relief="groove")
        self.load_border.grid(row=0, column=0, sticky="nsew", padx=20, pady=5)

        self.load_button = tk.Button(self.load_border,
                                     text="Load Image",
                                     font=("Arial", 12),
                                     background="yellow",
                                     command=self.load_image)
        self.load_button.pack(side=tk.TOP, padx=5, pady=5)

        # [3.3] Create a "Segment Image" button
        self.segment_border = tk.Frame(self.button_area1, relief="groove")
        self.segment_border.grid(row=0, column=1, sticky="nesw", padx=20, pady=5)

        self.segment_button = tk.Button(self.segment_border,
                                        text="Target Locate",
                                        font=("Arial", 12),
                                        background="yellow",
                                        command=self.target_locate)
        self.segment_button.pack(side=tk.TOP, padx=5, pady=5)

        # [3.4] Setup lower threshold bar for image segmentation
        self.min_area_border = tk.Frame(self.button_area1, relief="groove")
        self.min_area_border.grid(row=1, column=3, sticky="nsew")

        # set default value for lower threshold
        self.min_area = 20
        min_area = tk.IntVar()
        min_area.set(self.min_area)

        self.min_area_slider = tk.Scale(self.min_area_border, from_=5, to=100,
                                        label="Min Area",
                                        font=("Arial", 12),
                                        length=300,
                                        width=20,
                                        # showvalue=False,
                                        orient=tk.HORIZONTAL,
                                        variable=min_area,
                                        command=self.on_min_area_change)

        self.min_area_slider.pack(side=tk.TOP, padx=5, pady=5)


        # [3.5] Setup upper threshold bar for image segmentation
        self.max_area_border = tk.Frame(self.button_area1, relief="groove")
        self.max_area_border.grid(row=2, column=3, sticky="nesw", padx=20, pady=5)

        # set default value for upper threshold
        self.max_area = 450
        max_area = tk.IntVar()
        max_area.set(self.max_area)

        self.max_area_slider = tk.Scale(self.max_area_border, from_=150, to=550,
                                        label="Max Area",
                                        font=("Arial", 12),
                                        length=300,
                                        width=20,
                                        # showvalue=False,
                                        orient=tk.HORIZONTAL,
                                        variable=max_area,
                                        command=self.on_max_area_change)

        self.max_area_slider.pack(side=tk.TOP, padx=5, pady=5)



        # [3.6] Setup lower threshold bar for image segmentation
        self.min_thresold_border = tk.Frame(self.button_area1, relief="groove")
        self.min_thresold_border.grid(row=1, column=0, sticky="nsew")

        # set default value for lower threshold
        self.min_thresh = 50
        min_thresh = tk.IntVar()
        min_thresh.set(self.min_thresh)

        self.min_threshold_slider = tk.Scale(self.min_thresold_border, from_=0, to=100,
                                             label="Min threshold",
                                             font=("Arial", 12),
                                             length=300,
                                             width=20,
                                             # showvalue=False,
                                             orient=tk.HORIZONTAL,
                                             variable=min_thresh,
                                             command=self.on_min_threshold_change)

        self.min_threshold_slider.pack(side=tk.TOP, padx=5, pady=5)


        # [3.7] Setup upper threshold bar for image segmentation
        self.diff_threshold_border = tk.Frame(self.button_area1, relief="groove")
        self.diff_threshold_border.grid(row=2, column=0, sticky="nesw", padx=20, pady=5)

        # set default value for upper threshold
        self.diff_thresh = 50
        diff_thresh = tk.IntVar()
        diff_thresh.set(self.diff_thresh)

        self.ellips_threshold_slider = tk.Scale(self.diff_threshold_border, from_=0, to=180,
                                                label="Different threshold",
                                                font=("Arial", 12),
                                                length=300,
                                                width=20,
                                                # showvalue=False,
                                                orient=tk.HORIZONTAL,
                                                variable=diff_thresh,
                                                command=self.on_diff_threshold_change)

        self.ellips_threshold_slider.pack(side=tk.TOP, padx=5, pady=5)


        # [3.8] Setup lower threshold bar for image segmentation
        self.axis_ration_thresold_border = tk.Frame(self.button_area1, relief="groove")
        self.axis_ration_thresold_border.grid(row=3, column=0, sticky="nsew")

        # set default value for lower threshold
        self.axis_ratio_threshold = 0.3
        axis_ratio_threshold = tk.DoubleVar()
        axis_ratio_threshold.set(self.axis_ratio_threshold)

        self.axis_ratio_threshold_slider = tk.Scale(self.axis_ration_thresold_border, from_=0, to=1.0,
                                                    resolution=0.01,
                                                    label="Axis ratio threshold",
                                                    font=("Arial", 12),
                                                    length=300,
                                                    width=20,
                                                    # showvalue=False,
                                                    orient=tk.HORIZONTAL,
                                                    variable=axis_ratio_threshold,
                                                    command=self.on_axis_ratio_change)

        self.axis_ratio_threshold_slider.pack(side=tk.TOP, padx=5, pady=5)


        # [3.9] Setup upper threshold bar for image segmentation
        self.ellips_threshold_border = tk.Frame(self.button_area1, relief="groove")
        self.ellips_threshold_border.grid(row=3, column=3, sticky="nesw", padx=20, pady=5)

        # set default value for upper threshold
        self.ellips_threshold = 0.1
        ellips_threshold = tk.DoubleVar()
        ellips_threshold.set(self.ellips_threshold)

        self.ellips_threshold_slider = tk.Scale(self.ellips_threshold_border, from_=0, to=1.0,
                                                label="ellips threshold",
                                                resolution=0.01,
                                                font=("Arial", 12),
                                                length=300,
                                                width=20,
                                                # showvalue=False,
                                                orient=tk.HORIZONTAL,
                                                variable=ellips_threshold,
                                                command=self.on_ellips_threshold_change)

        self.ellips_threshold_slider.pack(side=tk.TOP, padx=5, pady=5)

    # Callback function when the lower threshold slider is changed
    def on_min_area_change(self, val):
        self.min_area = int(val)
        self.target_locate()

    # Callback function when the upper threshold slider is changed
    def on_max_area_change(self, val):
        self.max_area = int(val)
        self.target_locate()


    # Callback function when the upper threshold slider is changed
    def on_min_threshold_change(self, val):
        self.min_thresh = int(val)
        self.target_locate()

    # Callback function when the upper threshold slider is changed
    def on_diff_threshold_change(self, val):
        self.diff_thresh = int(val)
        self.target_locate()

    def on_axis_ratio_change(self, val):
        self.axis_ratio_threshold = float(val)
        self.target_locate()

    def on_ellips_threshold_change(self, val):
        self.ellips_threshold = float(val)
        self.target_locate()



    """
    The major function of this algorithm
    1. according to the upper and lower thresholds, execute the image segmentation in HSV space
    2. show the mask image in the right image showing area
    """
    def target_locate(self):

        test_img = np.array(self.original_image)
        # get the M
        mask = rough_target_detection(test_img, self.min_thresh, self.diff_thresh)
        # filter unwanted components that are too small or too big
        area_filtered_mask = area_threshold_filter(mask, self.min_area, self.max_area)
        # filter unwanted components that are obviously not round
        axis_ratio_mask = axis_ratio_filter(area_filtered_mask, self.axis_ratio_threshold)
        # selected targets clusters from targets mask
        targets_mask, cluster_labels, centroids, targets_list = target_detection(axis_ratio_mask, self.ellips_threshold)


        # Resize the image to fit in the label
        height, width = targets_mask.shape
        max_size = 500
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = int(width * (max_size / height))
            new_height = max_size
        resize_image = cv2.resize(targets_mask, (new_width, new_height))


        # Convert the image to Tkinter format and display it on the right side
        seg_photo = ImageTk.PhotoImage(image=Image.fromarray(resize_image))
        self.processed_img_canvas.create_image(0, 0, anchor=tk.NW, image=seg_photo)
        self.processed_img_canvas.image = seg_photo


    def load_image(self):
        # Open a file selection dialog box to choose an image file
        file_path = filedialog.askopenfilename(title="Select Image File",
                                               filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])

        # Load the chosen image using PIL
        self.original_image = Image.open(file_path)

        # Resize the image to fit in the label
        width, height = self.original_image.size
        max_size = 500
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = int(width * (max_size / height))
            new_height = max_size
        resize_image = self.original_image.resize((new_width, new_height))

        # Convert the image to Tkinter format and display it on the left side
        photo = ImageTk.PhotoImage(resize_image)
        self.orig_img_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.orig_img_canvas.image = photo


if __name__ == "__main__":
    root = tk.Tk()

    gui = ImageGUI(root)
    root.mainloop()
