import os
import cv2
import numpy as np
import keras_ocr
import pickle

def segment_image_info(image_name, image_path, label, images_dict, saved_folder_path, pipeline):
    """
    :param image_name: image name with extension
    :param image_path: image path
    :param label: G/G_Suspects1/G_Suspects2/S/S_Suspects1
    :param images_dict: sub-image info to be stored
    :param saved_folder_path: path to store the sub-images
    :param pipeline: used for keras OCR
    :return: None
    """
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
    """
    Segment the image and return the info of the sub-images
    :param removed_text_image: text-removed image
    :param original_image: original image
    :param name: image name w/o extension
    :param sub_image_dict: dict that store the sub-images' info
    :return: None
    """
    sub_image_names = ['En-face_52.0micrometer_Slab_(Retina_View)', 'Circumpapillary_RNFL',
                       'RNFL_Thickness_(Retina_View)', 'GCL+_Thickness_(Retina_View)',
                       'RNFL_Probability_and_VF_Test_points(Field_View)','GCL+_Probability_and_VF_Test_points']
    # Convert the image to grayscale
    gray = cv2.cvtColor(removed_text_image, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection
    edges = cv2.Canny(gray, 30, 80)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (this will help in avoiding noise)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 80000]
    ordered_contours = []

    # Get bounding box for each contour
    for contour in filtered_contours:
        ordered_contours.append(cv2.boundingRect(contour))

    ordered_contours = sorted(ordered_contours, key=lambda box: box[0])

    if len(filtered_contours) != 6:
        print('Segmentation failed: ' + name)

    # Loop through the contours and save the sub-images
    for index, contour in enumerate(ordered_contours):
        # Get bounding box for each contour
        x, y, w, h = ordered_contours[index]

        # Extract the sub-image using slicing
        sub_image = original_image[y:y + h, x:x + w]

        #For RLS images, find the top right cornor by detecting its red bounding box
        if index == 4 and name[0:3] == 'RLS':
            sub_image, x2, y2, w, h = RLS_red_box_detector(sub_image)
            x += x2
            y += y2
        # For non-RLS images, find the top right cornor by matching the white circle
        # with the white circle in RNFL_Thickness_(Retina_View)
        if index == 4 and name[0:3] != 'RLS':
            x_t, y_t, w_t, h_t = ordered_contours[2]
            template_image = original_image[y_t:y_t + h_t, x_t:x_t + w_t]
            x, y, w, h = non_RLS_crop(sub_image, template_image, x, y)
            sub_image = original_image[y:y + h, x:x + w]

        #For RLS images that are down-sampled, recompute the sub-images
        height, width, _ = original_image.shape
        if width < 2000 and name[0:3] == 'RLS':
            if index == 2:
                _, _, w_t, h_t = ordered_contours[0]
                w = w_t
                h = h_t
                sub_image = original_image[y:y + h, x:x + w]
            if index == 5:
                #crop GCL+ probability according to red bounding
                sub_image, x2, y2, w, h = RLS_red_box_detector(sub_image)
                x += x2
                y += y2

                #crop GCL+ thickness based on GCL+ probability's w and h
                x_re, y_re = sub_image_dict[sub_image_names[3]]['position'][0]
                re_crop_sub_image = original_image[y_re:y_re + h - 3, x_re:x_re + w]
                sub_image_dict[sub_image_names[3]] = {'sub_image': re_crop_sub_image, 'position': [(x_re, y_re), (x_re + w, y_re), (x_re + w, y_re + h - 3), (x_re, y_re + h - 3)]}

        #store the sub-image info in a dict
        sub_image_info = {'sub_image': sub_image, 'position': [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]}
        sub_image_dict[sub_image_names[index]] = sub_image_info

def text_remover_keras(img_path, pipeline):
    """
    Remove text with keras OCR
    :param img_path:
    :param pipeline:
    :return: text-removed image
    """
    # read the image
    img = keras_ocr.tools.read(img_path)

    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize([img])

    height, width, _ = img.shape

    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x0 = int(x0)
        x1 = int(x1)
        y0 = int(y0)
        y3 = int(y3)

        #replace text with larger area for top right region only
        if y0 < 0.6 * height and x0 < 0.6 * width:
            img[y0 - 30:y3 + 30, x0 - 30: x1 + 30] = [255, 255, 255]
        else:
            img[y0 - 10:y3 + 10, x0 - 10: x1 + 10] = [255, 255, 255]
    return img

def RLS_red_box_detector(image):
    """
    detect red bounding box for RLS reports
    :param image:
    :return: cropped_image, x, y, w, h
    """

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

def non_RLS_crop(target_image, template_image, x_original, y_original):
    """
    Match the white circle in both RNFL_Probability and RNFL_Thickness as they have
    the same dimension (flip RNFL_Thickness)
    :param target_image: RNFL_Probability_and_VF_Test_points(Field_View)
    :param template_image: RNFL_Thickness_(Retina_View)
    :param x_original: original x of the target_image in the report
    :param y_original: original y of the target_image in the report
    :return: recomputed x, y, w, h
    """
    flipped_template = cv2.flip(template_image, 0)
    x_target, y_target = find_circle(target_image)
    x_template, y_template = find_circle(flipped_template)
    h, w, _ = template_image.shape
    return x_original + x_target - x_template, y_original + y_target - y_template, w, h

def find_circle(image):
    """
    Find the white circle in the image
    :param image:
    :return: position of the circle in the image
    """

    #convert to grayscale and threshold the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresholded_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

    gray_blurred = cv2.blur(thresholded_image, (4, 4))

    # Apply Hough transform to detect circles in the image
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=300, param2=14, minRadius=100,
                               maxRadius=130)

    if len(circles) > 1:
        print('Circle detection failed')
    circles = np.round(circles[0, :]).astype("int")

    #return the first circle, which should also be the only coircle detected if worked porperly
    return circles[0][0], circles[0][1]

if __name__ == '__main__':

    dataset_dir = r'/Users/kuangsun/Desktop/test_reports'
    processed_images_info = {}
    dirs = os.listdir(dataset_dir)

    #keras OCR pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    #address to save result
    dir_name = 'Processed_data'
    os.makedirs(dir_name, exist_ok=True)

    for dir in dirs:
        if(dir[0] == '.'):
            continue
        current_folder_path = os.path.join(dir_name, dir)
        os.makedirs(current_folder_path, exist_ok=True)
        root_dir = os.path.join(dataset_dir, dir)

        image_names = [file for file in os.listdir(root_dir) if file.endswith('.png') or file.endswith('.jpg')]

        current_folder_dict = {}

        for image_name in image_names:
            image_path = os.path.join(dataset_dir, dir, image_name)
            segment_image_info(image_name, image_path, dir, current_folder_dict, current_folder_path, pipeline)

        processed_images_info[dir] = current_folder_dict

    #save info dict as a pickle file
    with open('oct_reports_info.pickle', 'wb') as handle:
        pickle.dump(processed_images_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Segmentation done')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
