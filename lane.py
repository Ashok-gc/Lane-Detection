import cv2
import numpy as np

# Define the region of interest (ROI) for lane detection
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Define the function to draw the detected lane lines
def draw_lines(img, lines):
    img = np.copy(img)
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

# Define the function to process each frame from the webcam
def process_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection to the image
    edges = cv2.Canny(blur, 50, 150)
    
    # Define the region of interest for the lane detection
    height, width = img.shape[:2]
    roi_vertices = [(0, height), (width/2, height/2), (width, height)]
    roi = region_of_interest(edges, np.array([roi_vertices], np.int32))
    
    # Apply Hough Transform to detect the lane lines
    lines = cv2.HoughLinesP(roi, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)
    
    # Draw the detected lane lines on the image
    img_with_lines = draw_lines(img, lines)
    
    return img_with_lines

# Open the webcam
cap = cv2.VideoCapture('project_video.mp4')

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Process the frame
    processed_frame = process_image(frame)
    
    # Show the processed frame
    cv2.imshow('Lane Detection', processed_frame)
    
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
