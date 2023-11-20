import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np

if __name__ == "__main__":
    # Load the image
    image = cv2.imread('sub_1.jpg')
    #
    # height, width, _ = image.shape
    #
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # Apply a threshold
    # _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    #
    # # Find contours
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Draw white squares
    # for cnt in contours:
    #     # Get bounding box
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     cv2.rectangle(image, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 255, 255), -1)

    square = 10
    height, width, _ = image.shape

    while height > 730 or width > 970:

        cv2.rectangle(image, (0, 0), (square, square), (255, 255, 255), -1)
        cv2.rectangle(image, (width - square, 0), (width, square), (255, 255, 255), -1)
        cv2.rectangle(image, (0, height - square), (square, height), (255, 255, 255), -1)
        cv2.rectangle(image, (width - square, height - square), (width, height), (255, 255, 255), -1)

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
        image = image[y:y + h, x:x + w]

        # plt.imshow(image)
        # plt.show()
        height_new, width_new, _ = image.shape
        if abs(height - height_new) < 2 and abs(width_new - width) < 2:
            break
        else:
            height, width, _ = image.shape

    print('done')


    cv2.imwrite('cropped_image_final.jpg', image)

