import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('lanes_clip.mp4')

# Define the region of interest
roi_vertices = [(0, 720), (1280, 720), (750, 460), (550, 460)]
def region_of_interest(img, vertices):
    # mask = np.zeros_like(img)
    # match_mask_color = (255,) * img.shape[2]
    # cv2.fillPoly(mask, [vertices], match_mask_color)
    # masked_image = cv2.bitwise_and(img, mask)
    # return masked_image
    # Define a blank mask to start with
    mask = np.zeros_like(img)

    # Determine the mask color based on the image type
    if len(img.shape) > 2:  # Colored image
        match_mask_color = (255,) * img.shape[2]
    else:  # Grayscale image
        match_mask_color = 255

    # Fill the polygon with white
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Mask the image
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img

# Define the line drawing function
def draw_lines(img, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, y_intercept))
        else:
            right_fit.append((slope, y_intercept))

    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    left_line = calculate_line_coordinates(img, left_fit_avg)
    right_line = calculate_line_coordinates(img, right_fit_avg)

    cv2.line(img, (int(left_line[0]), int(left_line[1])), (int(left_line[2]), int(left_line[3])), (0, 255, 0), 10)
    cv2.line(img, (int(right_line[0]), int(right_line[1])), (int(right_line[2]), int(right_line[3])), (0, 255, 0), 10)

def calculate_line_coordinates(img, parameters):
    # slope, y_intercept = parameters
    # y1 = img.shape[0]
    # y2 = int(y1 * 0.6)
    # x1 = int((y1 - y_intercept) / slope)
    # x2 = int((y2 - y_intercept) / slope)
    # return np.array([x1, y1, x2, y2])
    if parameters is None:
        return None
    elif len(parameters) != 2:
        return None
    else:
        slope, y_intercept = parameters
        y1 = img.shape[0]
        y2 = int(y1 * 0.6)
        x1 = int((y1 - y_intercept) / slope)
        x2 = int((y2 - y_intercept) / slope)
        return np.array([x1, y1, x2, y2])

# Process each frame
while True:
    # Capture the frame
    ret, frame = cap.read()

    # Apply a Gaussian blur to the image
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert the image to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Perform edge detection using Canny
    edges = cv2.Canny(gray, 50, 150)

    # Apply the region of interest mask
    roi = region_of_interest(edges, np.array([roi_vertices], np.int32))

    # Perform line detection using HoughLinesP
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, np.array([]), minLineLength=50, maxLineGap=5)

    # Draw the detected lines on the frame
    draw_lines(frame, lines)

    # Show the processed frame
    cv2.imshow('Lane Detection', frame)

    # Check if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
