import cv2
import numpy as np

img = cv2.imread('car.jpg')

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply edge detection method on image
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# returs array of r and theta values
lines = cv2.HoughLines(edges, 1, np.pi/180, 130)

# run loop until r and theta values are in the range of 2d array
for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr

    # store values of cos(theta) in a
    a = np.cos(theta)

    # store value of sin(theta) in b
    b = np.sin(theta)

    # x0 stores values rcos(theta)
    x0 = a*r

    # y0 stores the value rsin(theta)
    y0 = b*r

    # rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000*(-b))

    # rounded value of (rsin(theta)+ 1000cos(theta))
    y1 = int(y0 + 1000*(a))

    # x2 rounded for (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000*(-b))

    # y2 stores rounded for (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000*(a))

    # cv2.line draw line in img from the point (x1,y1) to (x2,y2) and color
    cv2.line(img, (x1,y1), (x2, y2), (0,0,255),2)

# all changes in input are written on new image
cv2.imwrite('new.jpg', img)

