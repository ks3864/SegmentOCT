# https://youtu.be/3RNPJbUHZKs
"""
Remove text from images

"""

import matplotlib.pyplot as plt
import easyocr
import cv2
import math
import numpy as np


# General Approach.....
# Use keras OCR to detect text, define a mask around the text, and inpaint the
# masked regions to remove the text.
# To apply the mask we need to provide the coordinates of the starting and
# the ending points of the line, and the thickness of the line

# The start point will be the mid-point between the top-left corner and
# the bottom-left corner of the box.
# the end point will be the mid-point between the top-right corner and the bottom-right corner.
# The following function does exactly that.
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


# Main function that detects text and inpaints.
# Inputs are the image path and kreas_ocr pipeline
def inpaint_text(img_path, reader):

    # # Read the image
    image = cv2.imread(img_path)
    #
    # # Preprocess the image by resizing
    scale_percent = 200  # scale by 200% to make the text larger
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    #
    # # Convert to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Define a dilation kernel. The size of the kernel affects the amount of dilation.
    # A larger kernel size will result in more dilation.
    kernel = np.ones((3, 3), np.uint8)

    # Apply dilation
    dilated_image = cv2.dilate(gray_image, kernel, iterations=1)

    # Detect text
    results = reader.detect(dilated_image, min_size=0, text_threshold=0, canvas_size=11000)

    # Remove text from the image
    image_without_text = remove_text(resized_image, results)

    width = int(image.shape[1] / (scale_percent / 100))
    height = int(image.shape[0] / (scale_percent / 100))
    dim = (width, height)
    final_image = cv2.resize(image_without_text, dim, interpolation=cv2.INTER_CUBIC)
    return final_image

def remove_text(img, detections):
    for detection in detections[0][0]:  # detections[0] contains the bounding boxes
        x_min = detection[0]
        x_max = detection[1]
        y_min = detection[2]
        y_max = detection[3]
        img[y_min-5:y_max+5, x_min-5:x_max+5] = [255, 180, 0]
    return img
#
# Create a reader to detect text
reader = easyocr.Reader(['en'], gpu=False)  # Specify the language(s)
img_text_removed = inpaint_text('RLS_127_OS_TC.jpg', reader)
#
plt.imshow(img_text_removed)
plt.show()
cv2.imwrite('text_removed_image14.jpg', cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))