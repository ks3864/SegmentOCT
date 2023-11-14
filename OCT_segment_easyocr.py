import os
import cv2
import numpy as np
import easyocr
import pickle
from matplotlib import pyplot as plt

def segment_image_info(image_name, image_path, label, images_dict, saved_folder_path, reader):
    # Load the image

    #sub_image_contours_pos = []
    sub_image_dict = {}

    image = cv2.imread(image_path)

    return_dict = {'Original_image': {'image_name': image_name, 'image': image}}

    image_copy = cv2.imread(image_path)
    removed_text_image = text_remover_easyocr(image_path, reader)

    name, ext = os.path.splitext(image_name)

    segment(removed_text_image, image, sub_image_dict)

    sub_images_path = os.path.join(saved_folder_path, name)
    os.makedirs(sub_images_path, exist_ok=True)
    for key in sub_image_dict:
        cv2.imwrite(os.path.join(sub_images_path, key + '.jpg'), sub_image_dict[key]['sub_image'])

    images_dict.update({name: {'original_image': image, 'sub_images': sub_image_dict}, 'label': label})

    print('Done')

def segment(removed_text_image, original_image, sub_image_dict):
    #sub_image_names = ['GCL+_Probability_and_VF_Test_points', 'GCL+_Thickness_(Retina_View)',
    #                   'RNFL_Thickness_(Retina_View)', 'En-face_52.0micrometer_Slab_(Retina_View)',
    #                   'RNFL_Probability_and_VF_Test_points(Field_View)', 'Circumpapillary_RNFL']
    sub_image_names = ['1', '2', '3', '4', '5', '6']
    # Convert the image to grayscale
    gray = cv2.cvtColor(removed_text_image, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    #edges = cv2.Canny(blurred, 30, 80)
    edges = cv2.Canny(gray, 30, 80)

    # plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (this will help in avoiding noise)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 80000]

    if len(filtered_contours) != 6:
        print('Segmentation failed:')

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

def text_remover_easyocr(img_path, reader):

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
    #
    # Apply thresholding
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Define a dilation kernel. The size of the kernel affects the amount of dilation.
    # A larger kernel size will result in more dilation.
    kernel = np.ones((3, 3), np.uint8)

    # Apply dilation
    dilated_image = cv2.dilate(gray_image, kernel, iterations=2)

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
        img[y_min-10:y_max+10, x_min-10:x_max+10] = [255, 255, 255]
    return img

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dataset_dir = r'C:\Users\Kuang Sun\Desktop\reports_cleaned'
    processed_images_info = {}
    dirs = os.listdir(dataset_dir)
    reader = easyocr.Reader(['en'], gpu=False)

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
            segment_image_info(image_name, image_path, dir, current_folder_dict, current_folder_path, reader)

        processed_images_info[dir] = current_folder_dict
    print('Segmentation done')

