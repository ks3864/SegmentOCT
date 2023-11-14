import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_text_from_image(img_path, output_path):
    # Read the image
    image = cv2.imread(img_path, 1)

    # Convert the image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to differentiate text from background
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours (text regions)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and fill the contours
    for contour in contours:
        # This can be adjusted depending on the size of the text
        if cv2.contourArea(contour) < 1000:
            x, y, w, h = cv2.boundingRect(contour)
            image[-3+y:y+h+3, -3+x:x+w+3] = (255, 255, 255)  # Fill with white color

    # Save the processed image
    cv2.imwrite(output_path, image)


def remove_text_from_image_new(image_path, output_path, threshold=127):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale for thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binarize
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Dilate the text region a bit
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Perform inpainting
    inpainted = cv2.inpaint(image, dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    return inpainted


if __name__ == "__main__":
    a = remove_text_from_image_new('oct_report1.png', 'output_image.png')
    # remove_text_from_image('oct_report1.png', 'output_image.png')

