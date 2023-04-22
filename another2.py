# import numpy as np
# import cv2

# def region_of_interest(img, vertices):
#     """
#     Applies an image mask.

#     Only keeps the region of the image defined by the polygon
#     formed from `vertices`. The rest of the image is set to black.
#     """
#     mask = np.zeros_like(img)
#     if len(img.shape) > 2:
#         channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
#         ignore_mask_color = (255,) * channel_count
#     else:
#         ignore_mask_color = 255
#     cv2.fillPoly(mask, vertices, ignore_mask_color)
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image

# def get_lane_parameters(lines):
#     """
#     Compute the average slope and intercept for a set of lines.
#     """
#     slopes = []
#     intercepts = []
#     lengths = []
#     for line in lines:
#         x1, y1, x2, y2 = line.reshape(4)
#         slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 999
#         intercept = y1 - slope * x1
#         length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
#         slopes.append(slope)
#         intercepts.append(intercept)
#         lengths.append(length)
#     # Use weighted average to compute the average slope and intercept,
#     # weighted by the line length.
#     avg_slope = np.average(slopes, weights=lengths)
#     avg_intercept = np.average(intercepts, weights=lengths)
#     return avg_slope, avg_intercept, sum(lengths)

# def draw_lanes(img, left_slope, left_intercept, right_slope, right_intercept):
#     """
#     Draw the left and right lane lines on an image.
#     """
#     y1 = img.shape[0]
#     y2 = int(y1 * 0.6)
#     left_x1 = int((y1 - left_intercept) / left_slope) if abs(left_slope) > 0.1 else 0
#     left_x2 = int((y2 - left_intercept) / left_slope) if abs(left_slope) > 0.1 else 0
#     right_x1 = int((y1 - right_intercept) / right_slope) if abs(right_slope) > 0.1 else 0
#     right_x2 = int((y2 - right_intercept) / right_slope) if abs(right_slope) > 0.1 else 0
#     cv2.line(img, (left_x1, y1), (left_x2, y2), (0, 0, 255), 10)
#     cv2.line(img, (right_x1, y1), (right_x2, y2), (0, 0, 255), 10)

# # Define the parameters for Canny edge detection and Hough transform
# canny_low_threshold = 50
# canny_high_threshold = 150
# hough_rho = 1
# hough_theta = np.pi/180
# hough_threshold = 30
# hough_min_line_length = 100
# hough_max_line_gap = 160

# # Define the moving average window size and initialize the lane line parameters
# window_size = 10
# left_slope_history = []
# left_intercept_history = []
# right_slope_history = []
# right_intercept_history = []

# def process_image(image):
#     """
#     Process an image to detect lane lines.
#     """
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     # Apply Gaussian blur
#     blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     # Apply Canny edge detection
#     edges = cv2.Canny(blur_gray, canny_low_threshold, canny_high_threshold)
#     # Define a four-sided polygon to mask
#     imshape = image.shape
#     vertices = np.array([[(0, imshape[0]), (imshape[1] / 2 - 45, imshape[0] / 2 + 60),
#                           (imshape[1] / 2 + 45, imshape[0] / 2 + 60), (imshape[1], imshape[0])]],
#                         dtype=np.int32)
#     # Apply the mask
#     masked_edges = region_of_interest(edges, vertices)
#     # Run Hough on edge detected image
#     lines = cv2.HoughLinesP(masked_edges, hough_rho, hough_theta, hough_threshold,
#                             np.array([]), hough_min_line_length, hough_max_line_gap)
#     # Separate the lines into left and right lines based on their slope
#     left_lines = []
#     right_lines = []
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 999
#             if slope < 0:
#                 left_lines.append(line)
#             else:
#                 right_lines.append(line)
#     # Compute the average slope and intercept for the left and right lines
#     left_slope, left_intercept, left_length = get_lane_parameters(left_lines)
#     right_slope, right_intercept, right_length = get_lane_parameters(right_lines)
#     # Compute the moving average of the lane line parameters
#     left_slope_history.append(left_slope)
#     left_intercept_history.append(left_intercept)
#     right_slope_history.append(right_slope)
#     right_intercept_history.append(right_intercept)
#     if len(left_slope_history) > window_size:
#         left_slope_history.pop(0)
#         left_intercept_history.pop(0)
#         right_slope_history.pop(0)
#         right_intercept_history.pop(0)
#     avg_left_slope = np.mean(left_slope_history)
#     avg_left_intercept = np.mean(left_intercept_history)
#     avg_right_slope = np.mean(right_slope_history)
#     avg_right_intercept = np.mean(right_intercept_history)
#     # Draw the left and right lane lines on the image
#     draw_lanes(image, avg_left_slope, avg_left_intercept, avg_right_slope, avg_right_intercept)
#     return image


# # Start the video capture
# cap = cv2.VideoCapture('project_video.mp4')
# while(cap.isOpened()):
#     # Read a frame from the video capture
#     ret, frame = cap.read()
#     if ret:
#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blur to smooth the image and reduce noise
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)

#         # Apply Canny edge detection to find the edges in the image
#         edges = cv2.Canny(blur, canny_low_threshold, canny_high_threshold)

#         # Define the region of interest as a trapezoid and mask the edges image
#         height, width = edges.shape
#         vertices = np.array([[(0, height), (width * 0.45, height * 0.6), 
#                               (width * 0.55, height * 0.6), (width, height)]], dtype=np.int32)
#         masked_edges = region_of_interest(edges, vertices)

#         # Apply Hough transform to find the line segments in the masked edges image
#         lines = cv2.HoughLinesP(masked_edges, hough_rho, hough_theta, hough_threshold, np.array([]),
#                                 minLineLength=hough_min_line_length, maxLineGap=hough_max_line_gap)

#         # Compute the parameters of the left and right lane lines using the line segments
#         left_lines = []
#         right_lines = []
#         for line in lines:
#             x1, y1, x2, y2 = line.reshape(4)
#             if x2 != x1:
#                 slope = (y2 - y1) / (x2 - x1)
#                 if slope < -0.5:
#                     left_lines.append(line)
#                 elif slope > 0.5:
#                     right_lines.append(line)
#         if len(left_lines) > 0:
#             left_slope, left_intercept, _ = get_lane_parameters(left_lines)
#             left_slope_history.append(left_slope)
#             left_intercept_history.append(left_intercept)
#         if len(right_lines) > 0:
#             right_slope, right_intercept, _ = get_lane_parameters(right_lines)
#             right_slope_history.append(right_slope)
#             right_intercept_history.append(right_intercept)

#         # If the moving average window is full, remove the oldest element
#         if len(left_slope_history) > window_size:
#             left_slope_history.pop(0)
#             left_intercept_history.pop(0)
#             right_slope_history.pop(0)
#             right_intercept_history.pop(0)

#         # Compute the moving average of the lane line parameters
#         left_slope_avg = np.mean(left_slope_history)
#         left_intercept_avg = np.mean(left_intercept_history)
#         right_slope_avg = np.mean(right_slope_history)
#         right_intercept_avg = np.mean(right_intercept_history)

#         # Draw the left and right lane lines on the frame
#         lanes = np.zeros_like(frame)
#         draw_lanes(lanes, left_slope_avg, left_intercept_avg, right_slope_avg, right_intercept_avg)
#         result = cv2.addWeighted(frame, 0.8, lanes, 1, 0)

#         # Show the result
#         cv2.imshow('Lane Detection', result)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# # Release the video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()









