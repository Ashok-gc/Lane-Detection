import cv2
import numpy as np

# Read the image
img = cv2.imread('lane.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to remove noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection to find edges
edges = cv2.Canny(blur, 50, 150)

# Define the Region of Interest
mask = np.zeros_like(edges)
height, width = img.shape[:2]
roi = [(0, height), (width/2, height/2), (width, height)]
polygon = np.array([roi], np.int32)
cv2.fillPoly(mask, polygon, 255)

# Apply bitwise AND to mask the edges
masked_edges = cv2.bitwise_and(edges, mask)

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=20)

# Draw lines on the original image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show the output
cv2.imshow('Lane Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
