# https://youtu.be/3RNPJbUHZKs
"""
Remove text from images

"""

import matplotlib.pyplot as plt
import pytesseract
import cv2
import math
import numpy as np

pytesseract.pytesseract.tesseract_cmd=r'D:\\Tesseract\\tesseract.exe'
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
def inpaint_text(img_path):
    # Load the image using OpenCV
    image = cv2.imread(img_path)

    # Use pytesseract to get the bounding box coordinates of the text
    detections = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Iterate over each text detection
    for i in range(len(detections['text'])):
        # If confidence is high enough (e.g., 60), remove the text
        if int(detections['conf'][i]) > 95:
            # Get the bounding box coordinates
            x, y, w, h = (detections['left'][i], detections['top'][i], detections['width'][i], detections['height'][i])

            # Draw a rectangle over the text (color and thickness chosen to match the background)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 180, 0), -1)
            image[y:y+h, x:x+w] = [255, 180, 0]




        # x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        # x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        #
        # # For the line thickness, we will calculate the length of the line between
        # # the top-left corner and the bottom-left corner.
        # thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        #
        # # Define the line and inpaint
        # cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,
        #          thickness)
        # inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    return image


# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
#
img_text_removed = inpaint_text('RLS_127_OS_TC.jpg')
#
plt.imshow(img_text_removed)
plt.show()
cv2.imwrite('text_removed_image18.jpg', cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))