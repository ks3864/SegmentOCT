import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_black_dots_and_interpolate(img_path, output_path):
    # 1. Load the Image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Detect Black Dots
    _, thresholded = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV)  # Assuming dots are darker than intensity 50
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Adjust kernel size based on dot size
    dilated = cv2.dilate(thresholded, kernel, iterations=2)

    # 3. Masking
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # 4. Remove Black Dots (Set to White)
    img[mask == 255] = [255, 255, 255]

    # 5. Interpolation
    result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Save the result
    cv2.imwrite(output_path, result)


# Usage
#remove_black_dots_and_interpolate('oct_report1.jpg', 'output_image.jpg')
