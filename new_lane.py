# import cv2
# import numpy as np

# # Load the video
# cap = cv2.VideoCapture('lanes_clip.mp4')

# # Define the region of interest
# roi_vertices = [(0, 720), (1280, 720), (750, 460), (550, 460)]
# def region_of_interest(img, vertices):
#     mask = np.zeros_like(img)
#     match_mask_color = (255,) * img.shape[2]
#     cv2.fillPoly(mask, [vertices], match_mask_color)
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image

# # Define the line drawing function
# def draw_lines(img, lines):
#     left_fit = []
#     right_fit = []
#     for line in lines:
#         x1, y1, x2, y2 = line.reshape(4)
#         parameters = np.polyfit((x1, x2), (y1, y2), 1)
#         slope = parameters[0]
#         y_intercept = parameters[1]
#         if slope < 0:
#             left_fit.append((slope, y_intercept))
#         else:
#             right_fit.append((slope, y_intercept))

#     left_fit_avg = np.average(left_fit, axis=0)
#     right_fit_avg = np.average(right_fit, axis=0)

#     left_line = calculate_line_coordinates(img, left_fit_avg)
#     right_line = calculate_line_coordinates(img, right_fit_avg)

#     cv2.line(img, (int(left_line[0]), int(left_line[1])), (int(left_line[2]), int(left_line[3])), (0, 255, 0), 10)
#     cv2.line(img, (int(right_line[0]), int(right_line[1])), (int(right_line[2]), int(right_line[3])), (0, 255, 0), 10)

# def calculate_line_coordinates(img, parameters):
#     if parameters is None or isinstance(parameters, np.float64):
#         return None
#     elif len(parameters) != 2:
#         return None
#     else:
#         slope, y_intercept = parameters
#         y1 = img.shape[0]
#         y2 = int(y1 * 0.6)
#         x1 = int((y1 - y_intercept) / slope)
#         x2 = int((y2 - y_intercept) / slope)
#         return np.array([x1, y1, x2, y2])

# # Process each frame
# while True:
#     # Capture the frame
#     ret, frame = cap.read()

#     # Apply a Gaussian blur to the image
#     blur = cv2.GaussianBlur(frame, (5, 5), 0)

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

#     # Perform edge detection using Canny
#     edges = cv2.Canny(gray, 50, 150)

#     # Apply the region of interest mask
#     roi = region_of_interest(edges, np.array([roi_vertices], np.int32))

#     # Perform line detection using HoughLinesP
#     lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

#     # Draw the detected lines onto the original image
#     draw_lines(frame, lines)

#     # Show the resulting image
#     cv2.imshow('frame', frame)

#     # Exit the loop if the 'q' key is pressed
#     if cv2.waitKey(1) == ord('q'):
#         break

# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('lanes_clip.mp4')

# Define the region of interest
roi_vertices = [(0, 720), (1280, 720), (750, 460), (550, 460)]
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        match_mask_color = (255,) * img.shape[2]
    else:
        match_mask_color = 255
    cv2.fillPoly(mask, [vertices], match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

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

    if len(left_fit) > 0:
        left_fit_avg = np.average(left_fit, axis=0)
        if left_fit_avg is not None and len(left_fit_avg) == 2:
            left_line = calculate_line_coordinates(img, left_fit_avg)
            cv2.line(img, (int(left_line[0]), int(left_line[1])), (int(left_line[2]), int(left_line[3])), (0, 255, 0), 10)

    if len(right_fit) > 0:
        right_fit_avg = np.average(right_fit, axis=0)
        if right_fit_avg is not None and len(right_fit_avg) == 2:
            right_line = calculate_line_coordinates(img, right_fit_avg)
            cv2.line(img, (int(right_line[0]), int(right_line[1])), (int(right_line[2]), int(right_line[3])), (0, 255, 0), 10)

def calculate_line_coordinates(img, parameters):
    if parameters is None or isinstance(parameters, np.float64):
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

    # Apply a gray blur to the image
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Apply region of interest mask
    masked_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

    # Apply Hough line detection
    lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=150)

    # Draw the detected lines on the original image
    line_image = np.zeros_like(frame)
    if lines is not None:
        draw_lines(line_image, lines)

    # Combine the line image with the original image
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # Display the resulting image
    cv2.imshow('Lane Detection', result)

    # Wait for key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
