import os
import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt

def segment_image(image_name, lable, image_path, saved_folder, images_dict):
    # Load the image

    #sub_image_contours_pos = []
    sub_image_dict = {}

    image = cv2.imread(image_path)

    return_dict = {'Original_image': {'image_name': image_name, 'image': image}}

    image_copy = cv2.imread(image_path)

    removed_text_image = remove_text_from_image(image_copy)

    name, ext = os.path.splitext(image_name)

    segment(removed_text_image, image, name, sub_image_dict)

    images_dict.update({name: {'original_image': image, 'sub_images': sub_image_dict}, 'lable': lable})

    print('Done')

def segment(image, original_image, image_name, sub_image_dict):
    sub_image_names = ['GCL+_Probability_and_VF_Test_points', 'GCL+_Thickness_(Retina_View)',
                       'RNFL_Thickness_(Retina_View)', 'En-face_52.0\u03BCm_Slab_(Retina_View)',
                       'RNFL_Probability_and_VF_Test_points(Field_View)', 'Circumpapillary_RNFL']
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 30, 80)

    # plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (this will help in avoiding noise)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 80000]

    if len(filtered_contours) != 6:
        print('Segmentation failed')

    # new_folder_path =
    # os.mkdir(os.path.join(saved_folder, name))

    # Loop through the contours and save the sub-images
    for index, contour in enumerate(filtered_contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the sub-image using slicing
        sub_image = original_image[y:y + h, x:x + w]

        # sub_image_contours_pos.append([(x, y),(x + w, y), (x + w, y + h), (x, y + h)])
        sub_image_info = {'sub_image': sub_image, 'position': [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]}
        if len(filtered_contours) != 6:
            plt.imshow(sub_image)
            plt.show()
        sub_image_dict[sub_image_names[index]] = sub_image_info

        # Save the sub-image
        cv2.imwrite(f'{image_name}_{sub_image_names[index]}.jpg', sub_image)

def remove_text_from_image(image):

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
    return image

def process_images(dataset_dir, images_dict):

    dirs = os.listdir(dataset_dir)

    dir_name = 'Processed_data'
    # os.makedirs(dir_name, exist_ok=True)

    for dir in dirs:

        current_folder_path = os.path.join(dir_name, dir)
        # os.makedirs(current_folder_path, exist_ok=True)
        root_dir = os.path.join(dataset_dir, dir)

        image_names = [file for file in os.listdir(root_dir) if file.endswith('.png') or file.endswith('.jpg')]

        current_folder_dict = {}

        for image_name in image_names:
            image_path = os.path.join(dataset_dir, dir, image_name)
            segment_image(image_name, dir, image_path, current_folder_path, current_folder_dict)

        images_dict[dir] = current_folder_dict


def crop(image_name, x):
    image = cv2.imread(image_name)
    height, width, _ = image.shape

    left = x
    right = width - x
    top = x
    bottom = height - x

    cropped_img = image[top:bottom, left:right]
    name, ext = os.path.splitext(image_name)
    cv2.imwrite(f'{name}_crop.png', cropped_img)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dataset_dir = r'C:\Users\Kuang Sun\Desktop\reports_cleaned'
    processed_images_info = {}
    process_images(dataset_dir, processed_images_info)
    # sub_image_dict = {}
    #
    # image_path = 'text_removed_image.jpg'
    # image_name = 'text_removed_image.jpg'
    #
    # image = cv2.imread('oct_report1.png')
    #
    # return_dict = {'Original_image': {'image_name': image_name, 'image': image}}
    #
    # image_copy = cv2.imread(image_path)
    #
    # removed_text_image = remove_text_from_image(image_copy)
    #
    # name, ext = os.path.splitext(image_name)
    #
    # segment(removed_text_image, image, name, sub_image_dict)

    print('Done')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
