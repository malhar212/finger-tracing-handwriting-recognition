# CS 5330
# Malhar Mahant & Kruthika Gangaraju & Sriram Kodeeswaran
# Final Project: Handwriting gesture detection and recognition
import os

import cv2
import imutils
import mediapipe as mp
import csv
import numpy as np

from tr_ocr import MyTrOCRModel

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Drawing finger tracing
def draw_finger_tracing(img, canvas, window_preview, results, prev_x, prev_y, frame_count, fps, draw_enabled):
    tracking = True
    cx, cy = 0, 0
    for handLms in results.multi_hand_landmarks:
        # Get index finger tip landmark
        index_finger_landmark = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        h, w, c = img.shape

        # Finding the coordinates of the landmark
        cx, cy = int(index_finger_landmark.x * w), int(index_finger_landmark.y * h)

        # Creating a circle around each landmark
        cv2.circle(img, (cx, cy), 10, (0, 255, 0),
                   cv2.FILLED)
        cv2.circle(window_preview, (cx, cy), 10, (0, 255, 0),
                    cv2.FILLED)
        
        # Drawing the lines
        if draw_enabled and cx > 0 and cy > 0 and prev_x > 0 and prev_y > 0:
            # Calculate the distance between the current and previous finger positions
            dist = np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
            # If the finger has moved more than a certain distance, reset the counter and draw a line
            if dist > 3:
                frame_count = 0
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 0), 5)
                cv2.line(window_preview, (prev_x, prev_y), (cx, cy), (0, 0, 0), 5)
            else:
                # If the finger has not moved enough, increment the counter
                frame_count += 1

            # If the finger has not moved enough for more than 30 frames, stop drawing lines, stop tracking
            # if frame_count > fps:
            #     tracking = False
            #     cx, cy = 0, 0
    return cx, cy, frame_count, tracking


# Processing the input image
def process_image(img):
    # Flip for mirror image
    img = cv2.flip(img, 1)
    # Converting the input to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)

    # Returning the detected hands to calling function
    return results, img


def clearCanvas(image):
    return np.zeros(image.shape, dtype=np.uint8) + 255


def calc_contours(image, cluster=False):
    # Invert the image using bitwise not operation
    inverted_image = cv2.bitwise_not(image)
    # Convert the image to grayscale
    gray = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary image
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Calculate the bounding box that encloses all contours
    x_min, y_min = np.inf, np.inf
    x_max, y_max = -np.inf, -np.inf

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if cluster:
        for contour in contours:
            for point in contour:
                x, y = point[0]
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
        # Draw the bounding box on the image
        padding = 20
        cv2.rectangle(image, (x_min-padding, y_min-padding), (x_max+padding, y_max+padding), (0, 255, 0), 2)
    contour = max(contours, key=lambda x: cv2.contourArea(x))
    return contour, (x_min, y_min, x_max-x_min, y_max-y_min)


def crop_to_content(image):
    contour1, bounding_box_og = calc_contours(image, True)
    padding = 20

    # Compute the oriented bounding box
    rect = cv2.minAreaRect(contour1)
    print(rect)
    x_min, y_min, height, width = bounding_box_og
    cv2.rectangle(image, (x_min-padding, y_min-padding), (x_min+height+padding, y_min+width+padding), (255, 255, 255), 2)

    # Get the rotation angle
    angle = rect[2]

    # Rotate the entire image
    rows, cols = rect[0] # Center of bounding box
    obb_center = (int(rows) , int(cols))
    M = cv2.getRotationMatrix2D(obb_center, angle-90 if angle > 45 else angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, image.shape[1::-1], borderValue=(255, 255, 255))

    # Crop the rotated image to the bounding rectangle
    padding = 20
    contour, bounding_box = calc_contours(rotated_image, True)
    x_min, y_min, height, width = bounding_box
    cv2.rectangle(image, (x_min-padding, y_min-padding), (x_min+height+padding, y_min+width+padding), (255, 255, 255), 2)
    padding = 2
    x, y, w, h = bounding_box # cv2.boundingRect(contour)
    cropped = rotated_image[y-padding:y+h+2*padding, x-padding:x+w+2*padding]
    return cropped


def main():
    # Replace 0 with the video path to use a
    # pre-recorded video
    cap = cv2.VideoCapture(0)
    canvas_name = 'Canvas'
    cv2.namedWindow(canvas_name, cv2.WINDOW_AUTOSIZE)
    ocr_model = 's'# MyTrOCRModel(trained_checkpoint="iam_word_trained_1")
    prev_x, prev_y = 0, 0
    frame_count = 0
    finger_detect_delay_count = 0
    tracking = True
    first_run = True
    draw_enabled = True
    key = ord('d')
    while True:
        # Taking the input
        success, image = cap.read()
        if not success:
            print("Could not access webcam")
            break
        fps = cap.get(cv2.CAP_PROP_FPS)
        # image = imutils.resize(image, width=500, height=500)
        if first_run:
            # Here is code for Canvas setup
            canvas = clearCanvas(image)
            window_preview = canvas.copy()
            first_run = False

        results, image = process_image(image)
        if tracking:
            if results.multi_hand_landmarks:
                if finger_detect_delay_count < 5:
                    finger_detect_delay_count += 1
                # Enable drawing if the hand was detected for at least 5 frames and drawing mode is selected
                draw_enabled = finger_detect_delay_count >= 5 and (key == ord('d') or key == ord('e'))
                window_preview = canvas.copy()
                prev_x, prev_y, frame_count, tracking = draw_finger_tracing(image, canvas, window_preview, results,
                                                                            prev_x, prev_y, frame_count, fps,
                                                                            draw_enabled)
                if key == ord('d'):
                    cv2.putText(image,
                            f"Drawing: {draw_enabled}",
                            (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                if key == ord('e'):
                    cv2.putText(image,
                            f"Drawing: {draw_enabled}, Evaluation mode ",
                            (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                finger_detect_delay_count = 0
        # else:
        #     # Start re-tracking only after hand has exited then re-entered the frame
        #     if not results.multi_hand_landmarks:
        #         tracking = True
        cv2.putText(image,
                    f"Tracking: {tracking}",
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        # Displaying the output
        cv2.imshow("Hand tracker", image)
        cv2.imshow(canvas_name, window_preview)
        temp = key
        key = cv2.waitKey(1)
        # Implements toggle functionality
        if key == temp:
            # toggle drawing
            if key == ord('d') or key == ord('e'):
                # tracking = not tracking
                # draw_enabled = not draw_enabled
                finger_detect_delay_count = 0
            key = 0
        if key == -1:
            key = temp
        # Program terminates when q key is pressed
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

        if key == ord('s'):
            if temp == ord('d'):
                handle_image_save(canvas)
            if temp == ord('e'):
                preview = canvas.copy()
                preview = crop_to_content(preview)
                cv2.imshow("Evaluate", preview)
                string = ocr_model.predict(preview)
                print(string)
        if key == ord('c'):
            canvas = clearCanvas(image)
            window_preview = canvas.copy()
        # Avoid removal of filter by other keyinputs
        if key != 0 and key != ord('d') and key != ord('e'):
            key = temp


def handle_image_save(canvas):
    # Wait for label input
    previewImage = canvas.copy()
    label = ""
    while True:
        cv2.putText(previewImage,
                    "Enter the text in the image and hit 'Enter' or 'Esc' key for training.",
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(previewImage, "Label: " + label, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("Add Label", previewImage)
        d = cv2.waitKey(0)
        if d == 13:  # 'Enter' key
            folder_path = "data"
            # Check if the folder exists
            if not os.path.exists(folder_path):
                # If not, create the folder
                os.makedirs(folder_path)
            filename = f'{label}.jpg'
            count = 1
            while os.path.exists(folder_path + "/" +filename):
                # Append "_count" to the file name
                filename = f"{label}_{count}.jpg"
                count += 1
            with open("custom_data.csv", mode="a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([folder_path + "/" +filename, label])
            canvas = crop_to_content(canvas)
            cv2.imshow("Add Label", previewImage)
            cv2.imwrite(folder_path + "/" + filename, canvas)
            cv2.destroyWindow("Add Label")
            break
        elif d == 27:  # 'Esc' key
            cv2.destroyWindow("Add Label")
            break
        else:
            label += chr(d)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
