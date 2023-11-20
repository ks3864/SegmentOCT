import os
import cv2
import numpy as np
import keras_ocr

def segment_image_info(image_name, image_path, pipeline, result_txt, failed_counter, total_counter):

    removed_text_image = text_remover_keras(image_path, pipeline)

    total_counter += 1

    success = segment(removed_text_image)

    if(success==False):
        print('Segment failed: ' + image_name)
        failed_counter += 1
        # Open the file with write permission
        result_txt.write(str(failed_counter) + ': ' + image_name)

def segment(removed_text_image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(removed_text_image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 30, 80)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (this will help in avoiding noise)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 80000]

    if len(filtered_contours) != 6:
        return False
    return True
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
        img[y0-30:y3+30, x0-30: x1+30] = [255, 255, 255]
    return img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dataset_dir = r'C:\Users\Kuang Sun\Desktop\reports_cleaned'
    processed_images_info = {}
    dirs = os.listdir(dataset_dir)
    pipeline = keras_ocr.pipeline.Pipeline()

    # The name of the file you want to create and write to
    result_txt = open("result.txt", "a")

    failed_counter = 0
    total_counter = 0

    for dir in dirs:

        root_dir = os.path.join(dataset_dir, dir)

        image_names = [file for file in os.listdir(root_dir) if file.endswith('.png') or file.endswith('.jpg')]

        for image_name in image_names:
            image_path = os.path.join(dataset_dir, dir, image_name)
            segment_image_info(image_name, image_path, pipeline, result_txt, failed_counter, total_counter)

    result_txt.close()
    print('Process finished: ' + str(failed_counter) + '/' + str(total_counter))


