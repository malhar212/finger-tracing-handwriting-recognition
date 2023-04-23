# CS 5330
# Malhar Mahant & Kruthika Gangaraju & Sriram Kodeeswaran
# Final Project: Handwriting gesture detection and recognition
import os

import cv2
import imutils
import mediapipe as mp
import csv
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Drawing finger tracing
def draw_finger_tracing(img, canvas, results, prev_x, prev_y, frame_count, fps, draw_enabled):
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

        # Drawing the landmark connections
        # mpDraw.draw_landmarks(img, [{'landmark': index_finger_landmark}],
        #                      mpHands.HAND_CONNECTIONS)]
        # print(draw_enabled)
        # print(draw_enabled and cx > 0 and cy > 0 and prev_x > 0 and prev_y > 0)
        if draw_enabled and cx > 0 and cy > 0 and prev_x > 0 and prev_y > 0:
            # Calculate the distance between the current and previous finger positions
            dist = np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
            # If the finger has moved more than a certain distance, reset the counter and draw a line
            if dist > 3:
                frame_count = 0
                cv2.circle(canvas, (cx, cy), 1, (0, 0, 0),
                           cv2.FILLED)
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 0), 2)
            else:
                # If the finger has not moved enough, increment the counter
                frame_count += 1

            # If the finger has not moved enough for more than 30 frames, stop drawing lines, stop tracking
            if frame_count > fps / 3:
                tracking = False
                cx, cy = 0, 0
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
    return np.zeros(image.shape) + 255


def main():
    # Replace 0 with the video path to use a
    # pre-recorded video
    cap = cv2.VideoCapture(0)
    canvas_name = 'Canvas'
    cv2.namedWindow(canvas_name, cv2.WINDOW_AUTOSIZE)

    prev_x, prev_y = 0, 0
    frame_count = 0
    finger_detect_delay_count = 0
    tracking = True
    first_run = True
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
            first_run = False
        results, image = process_image(image)
        if tracking:
            if results.multi_hand_landmarks:
                if finger_detect_delay_count < 5:
                    finger_detect_delay_count += 1
                # Enable drawing if the hand was detected for at least 5 frames and drawing mode is selected
                draw_enabled = finger_detect_delay_count >= 5 and key == ord('d')
                prev_x, prev_y, frame_count, tracking = draw_finger_tracing(image, canvas, results,
                                                                            prev_x, prev_y, frame_count, fps,
                                                                            draw_enabled)
                cv2.putText(image,
                            f"Drawing: {draw_enabled}",
                            (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                finger_detect_delay_count = 0
        else:
            # Start re-tracking only after hand has exited then re-entered the frame
            if not results.multi_hand_landmarks:
                tracking = True
        cv2.putText(image,
                    f"Tracking: {tracking}",
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        # Displaying the output
        cv2.imshow("Hand tracker", image)
        cv2.imshow(canvas_name, canvas)
        temp = key
        key = cv2.waitKey(1)
        # Implements toggle functionality
        if key == temp:
            if key == ord('d'):
                tracking = True
            key = 0
        if key == -1:
            key = temp
        # Program terminates when q key is pressed
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

        if key == ord('s'):
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
                    filename = f'{label}.jpg'
                    count = 1
                    while os.path.exists(filename):
                        # Append "_count" to the file name
                        filename = f"{label}_{count}.jpg"
                        count += 1
                    with open("custom_data.csv", mode="a", newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([filename, label])
                    cv2.imwrite(filename, canvas)
                    cv2.destroyWindow("Add Label")
                    break
                elif d == 27:  # 'Esc' key
                    cv2.destroyWindow("Add Label")
                    break
                else:
                    label += chr(d)
        if key == ord('c'):
            canvas = clearCanvas(image)

        # Avoid removal of filter by other keyinputs
        if key != 0 and key != ord('d'):
            key = temp


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
