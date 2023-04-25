import cv2
import numpy as np

# Load the image
image = cv2.imread('test2.png')



def calc_contours(image, cluster=False):
    print(image.dtype)
    # Invert the image using bitwise not operation
    inverted_image = cv2.bitwise_not(image)
    cv2.imshow('Binary image', inverted_image)
    cv2.waitKey(0)
    # Convert the image to grayscale
    gray = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary image
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary image', thresh)
    cv2.waitKey(0)

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
        cv2.imshow('A', image)
        cv2.waitKey(0)
        cv2.destroyWindow('A')
    contour = max(contours, key=lambda x: cv2.contourArea(x))
    return contour, (x_min, y_min, x_max-x_min, y_max-y_min)

contour1, bounding_box_og = calc_contours(image, True)
padding = 20
# Compute the oriented bounding box
rect = cv2.minAreaRect(contour1)
print(rect)
box = cv2.boxPoints(rect)
box_og = np.int0(box)
x_min, y_min, height, width = bounding_box_og
cv2.rectangle(image, (x_min-padding, y_min-padding), (x_min+height+padding, y_min+width+padding), (255, 255, 255), 2)
# Draw the bounding box on the original image
# cv2.drawContours(image, [box_og], 0, (0, 0, 255), 2)
# cv2.imshow('Contours', image)
# cv2.waitKey(0)
# Get the rotation angle
angle = rect[2]

# Rotate the entire image
rows, cols = rect[0] # Center of bounding box
obb_center = (int(rows) , int(cols))
# cv2.circle(image, obb_center, 10, 255, 10)
# cv2.imshow("A", image)
# cv2.waitKey(0)
M = cv2.getRotationMatrix2D(obb_center, angle-90 if angle > 45 else angle, 1.0)
rotated_image = cv2.warpAffine(image, M, image.shape[1::-1], borderValue=(255, 255, 255))
# cv2.imshow("Rotated", rotated_image)
# cv2.waitKey(0)
# Crop the rotated image to the bounding rectangle
padding = 20
contour, bounding_box = calc_contours(rotated_image, True)
x_min, y_min, height, width = bounding_box
cv2.rectangle(image, (x_min-padding, y_min-padding), (x_min+height+padding, y_min+width+padding), (255, 255, 255), 2)
# rect = cv2.minAreaRect(contour)
# print(rect)
# box = cv2.boxPoints(rect)
# box = np.int0(box)

# Draw the bounding box on the original image
# cv2.drawContours(rotated_image, [box], 0, (255, 255, 0), 2)
# cv2.imshow("Rot Contours", rotated_image)
# cv2.waitKey(0)
# print(box)
padding = 2
x, y, w, h = bounding_box # cv2.boundingRect(contour)
cropped = rotated_image[y-padding:y+h+2*padding, x-padding:x+w+2*padding]
cv2.imshow("Cropped", cropped)
cv2.waitKey(0)
# Save new image
cv2.imwrite('cropped.png', cropped)