import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np

if __name__ == "__main__":
    # # Read the image
    # image = cv2.imread('sub_1.jpg')
    #
    # # Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # Threshold to get the black dots
    # # Adjust the threshold value '10' as needed to capture all black dots
    # _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    #
    # # Invert the mask to get the black dots as black on white background
    # mask_inv = cv2.bitwise_not(mask)
    #
    # # Make the black dots white in the original image
    # image[mask == 255] = (255, 255, 255)
    #
    # # Save or display the modified image
    # cv2.imwrite('cropped image.jpg', image)
    #
    # image = cv2.imread('cropped image.jpg')
    #
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # contours, hierarchy  = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    # cv2.imwrite('Contours.jpg', image)
    # # Find the largest contour and its bounding box
    # largest_contour = max(contours, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(largest_contour)
    #
    # # Crop the image using the bounding box coordinates
    # cropped_image = image[y:y + h, x:x + w]
    #
    # # Save or display the cropped image
    # cv2.imwrite('cropped_image_final.jpg', cropped_image)


    # Load the image
    image = cv2.imread('sub_1.jpg')

    height, width, _ = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw white squares
    for cnt in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(image, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 255, 255), -1)

    square = 20
    cv2.rectangle(image, (0, 0), (square, square), (255, 255, 255), -1)
    cv2.rectangle(image, (width - square, 0), (width, 50), (255, 255, 255), -1)
    cv2.rectangle(image, (0, height - square), (50, height), (255, 255, 255), -1)
    cv2.rectangle(image, (width - square, height - square), (width, height), (255, 255, 255), -1)


    # Save the result
    cv2.imwrite('dot_removed.jpg', image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 30, 80)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour and its bounding box
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Crop the image using the bounding box coordinates
    cropped_image = image[y:y + h, x:x + w]

    height, width, _ = cropped_image.shape
    while height > 730 and width > 970:

        cv2.rectangle(cropped_image, (0, 0), (square, square), (255, 255, 255), -1)
        cv2.rectangle(cropped_image, (width - square, 0), (width, 50), (255, 255, 255), -1)
        cv2.rectangle(cropped_image, (0, height - square), (50, height), (255, 255, 255), -1)
        cv2.rectangle(cropped_image, (width - square, height - square), (width, height), (255, 255, 255), -1)

        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(cropped_image, 30, 80)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the largest contour and its bounding box
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Crop the image using the bounding box coordinates
        cropped_image = cropped_image[y:y + h, x:x + w]

        plt.imshow(cropped_image)
        plt.show()
        height_new, width_new, _ = cropped_image.shape
        if abs(height - height_new) < 2 and abs(width_new - width) < 2:
            break
        else:
            height, width, _ = cropped_image.shape


    cv2.imwrite('cropped_image_final.jpg', cropped_image)

