import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np



if __name__ == "__main__":
    # Read the image
    image = cv2.imread('sub_1.jpg')

    # Convert from BGR to HSV colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV
    lower_red = np.array([0, 30, 50])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour and its bounding box
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image using the bounding box coordinates
    cropped_image = image[y:y + h, x:x + w]

    # Save or display the cropped image
    cv2.imwrite('cropped_image.jpg', cropped_image)
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


