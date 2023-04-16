# CS 5330
# Malhar Mahant & Kruthika Gangaraju & Sriram Kodeeswaran
# Final Project: Handwriting gesture detection and recognition

import cv2
import mediapipe as mp
import imutils
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Drawing landmark connections
def draw_finger_tracing(img, canvas, results, prev_x, prev_y, frame_count, fps):
    tracking = True
    cx, cy = 0, 0
    if results.multi_hand_landmarks:
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
            #                      mpHands.HAND_CONNECTIONS)
            if cx > 0 and cy > 0 and prev_x > 0 and prev_y > 0:
                # Calculate the distance between the current and previous finger positions
                dist = np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)

                # If the finger has moved more than a certain distance, reset the counter and draw a line
                if dist > 3:
                    frame_count = 0
                    cv2.circle(canvas, (cx, cy), 1, (0, 255, 0),
                               cv2.FILLED)
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 255, 0), 1)
                else:
                    # If the finger has not moved enough, increment the counter
                    frame_count += 1

                # If the finger has not moved enough for more than 30 frames, stop drawing lines, stop tracking
                if frame_count > fps / 3:
                    tracking = False
                    cx, cy = 0, 0
        return cx, cy, frame_count, tracking
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


def main():
    # Replace 0 with the video path to use a
    # pre-recorded video
    cap = cv2.VideoCapture(0)
    canvas_name = 'Canvas'
    cv2.namedWindow(canvas_name, cv2.WINDOW_AUTOSIZE)
    # Here is code for Canvas setup
    canvas = np.zeros((471, 636, 3)) + 255
    prev_x, prev_y = 0, 0
    frame_count = 0
    tracking = True
    while True:
        # Taking the input
        success, image = cap.read()
        if not success:
            print("Could not access webcam")
            break
        fps = cap.get(cv2.CAP_PROP_FPS)
        # image = imutils.resize(image, width=500, height=500)

        results, image = process_image(image)
        if tracking:
            prev_x, prev_y, frame_count, tracking = draw_finger_tracing(image, canvas, results,
                                                                        prev_x, prev_y, frame_count, fps)
        else:
            # Start re-tracking only after hand has exited then re-entered the frame
            if not results.multi_hand_landmarks:
                tracking = True

        # Displaying the output
        cv2.imshow("Hand tracker", image)
        cv2.imshow(canvas_name, canvas)
        # Program terminates when q key is pressed
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
