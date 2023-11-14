import os
import cv2
import numpy as np
import keras_ocr
import easyocr
import pickle
from matplotlib import pyplot as plt

def segment_image_info(image_name, image_path, label, images_dict, saved_folder_path, pipeline):
    # Load the image

    sub_image_dict = {}

    image = cv2.imread(image_path)

    removed_text_image = text_remover_keras(image_path, pipeline)

    name, ext = os.path.splitext(image_name)

    segment(removed_text_image, image, name, sub_image_dict)

    sub_images_path = os.path.join(saved_folder_path, name)
    os.makedirs(sub_images_path, exist_ok=True)
    result_image = image.copy()
    for key in sub_image_dict:
        cv2.imwrite(os.path.join(sub_images_path, key + '.jpg'), sub_image_dict[key]['sub_image'])
        start_point = sub_image_dict[key]['position'][0]
        end_point = sub_image_dict[key]['position'][2]
        cv2.rectangle(result_image, start_point, end_point, (233, 0, 255), 3)
    cv2.imwrite(os.path.join(sub_images_path,'0_result.jpg'), result_image)

    images_dict.update({name: {'original_image': image, 'sub_images': sub_image_dict, 'label': label}})

    print('Done')

def segment(removed_text_image, original_image, name, sub_image_dict):
    sub_image_names = ['En-face_52.0micrometer_Slab_(Retina_View)', 'Circumpapillary_RNFL',
                       'RNFL_Thickness_(Retina_View)', 'GCL+_Thickness_(Retina_View)',
                       'RNFL_Probability_and_VF_Test_points(Field_View)','GCL+_Probability_and_VF_Test_points']
    # Convert the image to grayscale
    gray = cv2.cvtColor(removed_text_image, cv2.COLOR_BGR2GRAY)


    if name[0:3] != 'RLS':
        remove_dots_topright(gray)

    # Use Canny edge detection
    edges = cv2.Canny(gray, 30, 80)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (this will help in avoiding noise)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 80000]
    ordered_contours = []

    for contour in filtered_contours:
        # Get bounding box for each contour
        ordered_contours.append(cv2.boundingRect(contour))

    ordered_contours = sorted(ordered_contours, key=lambda box: box[0])

    if len(filtered_contours) != 6:
        print('Segmentation failed')

    # Loop through the contours and save the sub-images
    for index, contour in enumerate(ordered_contours):
        # Get bounding box for each contour
        x, y, w, h = ordered_contours[index]

        # Extract the sub-image using slicing
        sub_image = original_image[y:y + h, x:x + w]

        if index == 4 and name[0:3] == 'RLS':
            sub_image, x2, y2, w, h = RLS_red_box_detector(sub_image)
            x += x2
            y += y2

        if index == 4 and name[0:3] != 'RLS':
            height, width, _ = sub_image.shape

            if height > 740 or width > 980:
                x, y, w, h = non_RLS_crop(sub_image, x, y)
                sub_image = original_image[y:y + h, x:x + w]




        sub_image_info = {'sub_image': sub_image, 'position': [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]}
        sub_image_dict[sub_image_names[index]] = sub_image_info
#
#
def text_remover_keras(img_path, pipeline):
    # read the image
    img = keras_ocr.tools.read(img_path)

    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize([img])

    # Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x0 = int(x0)
        x1 = int(x1)
        y0 = int(y0)
        y3 = int(y3)
        img[y0-10:y3+10, x0-10: x1+10] = [255, 255, 255]
    return img

def RLS_red_box_detector(image):

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
    return cropped_image, x, y, w, h

def remove_dots_topright(gray):

    height, width = gray.shape

    # Apply a threshold
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw white squares
    for cnt in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        if x > width * 0.6 and y < height * 0.7:
            # Draw a white rectangle over the black dot
            cv2.rectangle(gray, (x - 3, y - 3), (x + w + 3, y + h + 3), (255, 255, 255), -1)

def non_RLS_crop(original_image, x_original, y_original):

    image = original_image.copy()

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

    square = 40
    height, width, _ = image.shape

    while height > 725 and width > 965:

        cv2.rectangle(image, (0, 0), (square, square), (255, 255, 255), -1)
        cv2.rectangle(image, (width - square, 0), (width, 50), (255, 255, 255), -1)
        cv2.rectangle(image, (0, height - square), (50, height), (255, 255, 255), -1)
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
        x_original += x
        y_original += y
        image = image[y:y + h, x:x + w]
        height, width, _ = image.shape

    return x_original, y_original, width, height

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dataset_dir = r'C:\Users\Kuang Sun\Desktop\reports_cleaned'
    processed_images_info = {}
    dirs = os.listdir(dataset_dir)
    pipeline = keras_ocr.pipeline.Pipeline()

    dir_name = 'Processed_data'
    os.makedirs(dir_name, exist_ok=True)

    for dir in dirs:

        current_folder_path = os.path.join(dir_name, dir)
        os.makedirs(current_folder_path, exist_ok=True)
        root_dir = os.path.join(dataset_dir, dir)

        image_names = [file for file in os.listdir(root_dir) if file.endswith('.png') or file.endswith('.jpg')]

        current_folder_dict = {}

        for image_name in image_names:
            image_path = os.path.join(dataset_dir, dir, image_name)
            segment_image_info(image_name, image_path, dir, current_folder_dict, current_folder_path, pipeline)

        processed_images_info[dir] = current_folder_dict

    with open('oct_reports_info.pickle', 'wb') as handle:
        pickle.dump(processed_images_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Segmentation done')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
