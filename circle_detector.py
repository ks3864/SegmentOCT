import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = 'RNFL_Thickness_(Retina_View).jpg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_flipped = cv2.flip(image, 0)

gray_image = cv2.cvtColor(image_flipped, cv2.COLOR_BGR2GRAY)

_, thresholded_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

gray_blurred = cv2.blur(thresholded_image, (3, 3))

# Apply Hough transform to detect circles in the image
circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=300, param2=14, minRadius=110, maxRadius=120)

# If circles are detected, draw them
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image_flipped, (x, y), r, (255, 0, 0), 4)

# # Show the result
cv2.imwrite('detected_circle.jpg', image_flipped)
# circles[0] if circles is not None else "No circles detected"

plt.imshow(image_flipped)
plt.show()



