import cv2
import numpy as np

# Load image
img = cv2.imread('car.jpg')

# Draw a horizontal line at y=200
cv2.line(img, (0, 170), (img.shape[1], 170), (255, 0, 0), 2)  # Blue color, thickness 2
cv2.line(img, (130, 155), (img.shape[0], 250), (0, 255, 0), 2)  

# Save the modified image
cv2.imwrite('image_with_horizontal_line.jpg', img)

# Save the modified image
cv2.imwrite('new_with_custom_lines.jpg', img)


