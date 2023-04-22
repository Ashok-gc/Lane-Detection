# import cv2
# import numpy as np

# # Define the region of interest (ROI) for lane detection
# def region_of_interest(img, vertices):
#     mask = np.zeros_like(img)
#     match_mask_color = 255
#     cv2.fillPoly(mask, vertices, match_mask_color)
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image

# # Define the function to draw the detected lane lines
# def draw_lines(img, lines):
#     img = np.copy(img)
#     line_img = np.zeros_like(img)
#     if lines is not None:
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
#     img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
#     return img

# # Define the function to process each frame from the webcam
# def process_image(img):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to the image
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply Canny edge detection to the image
#     edges = cv2.Canny(blur, 50, 150)
    
#     # Define the region of interest for the lane detection
#     height, width = img.shape[:2]
#     roi_vertices = [(0, height), (width/2, height/2), (width, height)]
#     roi = region_of_interest(edges, np.array([roi_vertices], np.int32))
    
#     # Apply Hough Transform to detect the lane lines
#     lines = cv2.HoughLinesP(roi, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)
    
#     # Draw the detected lane lines on the image
#     img_with_lines = draw_lines(img, lines)
    
#     return img_with_lines

# # Open the webcam
# cap = cv2.VideoCapture('project_video.mp4')

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
    
#     # Process the frame
#     processed_frame = process_image(frame)
    
#     # Show the processed frame
#     cv2.imshow('Lane Detection', processed_frame)
    
#     # Exit if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Define the region of interest (ROI) for lane detection
# def region_of_interest(img, vertices):
#     mask = np.zeros_like(img)
#     match_mask_color = 255
#     cv2.fillPoly(mask, vertices, match_mask_color)
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image

# # Define the function to draw the detected lane lines
# def draw_lines(img, lines):
#     img = np.copy(img)
#     line_img = np.zeros_like(img)
#     if lines is not None:
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
#     img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
#     return img

# # Define the function to process each frame from the webcam
# def process_image(img):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to the image
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply Canny edge detection to the image
#     edges = cv2.Canny(blur, 50, 150)
    
#     # Define the region of interest for the lane detection
#     height, width = img.shape[:2]
#     roi_vertices = [(0, height), (width/2, height/2), (width, height)]
#     roi = region_of_interest(edges, np.array([roi_vertices], np.int32))
    
#     # Apply Hough Transform to detect the lane lines
#     lines = cv2.HoughLinesP(roi, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)
    
#     # Draw the detected lane lines on the image
#     img_with_lines = draw_lines(img, lines)
    
#     return img_with_lines

# # Open the webcam
# cap = cv2.VideoCapture('project_video.mp4')

# while cap.isOpened():
#     # Read a frame from the webcam
#     ret, frame = cap.read()
    
#     # Process the frame
#     processed_frame = process_image(frame)
    
#     # Show the processed frame
#     cv2.imshow('Lane Detection', processed_frame)
    
#     # Exit if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()






# # rough-----------------------------------------


# import cv2
# import numpy as np

# # Define the region of interest (ROI) for lane detection
# def region_of_interest(img, vertices):
#     mask = np.zeros_like(img)
#     match_mask_color = 255
#     cv2.fillPoly(mask, vertices, match_mask_color)
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image

# # Define the function to draw the detected lane lines
# # def draw_lines(img, lines):
# #     img = np.copy(img)
# #     line_img = np.zeros_like(img)
# #     if lines is not None:
# #         for line in lines:
# #             for x1, y1, x2, y2 in line:
# #                 if x1 < img.shape[1] / 2 and x2 < img.shape[1] / 2:
# #                     cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=5) # Left lane: Red
# #                 else:
# #                     cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), thickness=5) # Right lane: Blue
# #     img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
# #     return img
# def draw_lines(img, lines):
#     img = np.copy(img)
#     line_img = np.zeros_like(img)
#     if lines is not None:
#         # Separate the lines into left and right lane lines based on their slopes
#         left_lines, right_lines = [], []
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             if x2 - x1 == 0:  # Avoid dividing by zero
#                 continue
#             slope = (y2 - y1) / (x2 - x1)
#             if slope < 0:
#                 left_lines.append(line)
#             elif slope > 0:
#                 right_lines.append(line)

#         # Fit a line to the left lane lines using linear regression
#         left_x, left_y = [], []
#         for line in left_lines:
#             x1, y1, x2, y2 = line[0]
#             left_x += [x1, x2]
#             left_y += [y1, y2]
#         if len(left_x) > 0 and len(left_y) > 0:
#             left_fit = np.polyfit(left_x, left_y, 1)
#             left_slope, left_intercept = left_fit

#             # Extrapolate the left lane line to the bottom and top of the lane
#             left_y1 = img.shape[0]
#             left_x1 = int((left_y1 - left_intercept) / left_slope)
#             left_y2 = int(img.shape[0] * 0.6)
#             left_x2 = int((left_y2 - left_intercept) / left_slope)
#             cv2.line(line_img, (left_x1, left_y1), (left_x2, left_y2), (0, 255, 0), thickness=10)

#         # Fit a line to the right lane lines using linear regression
#         right_x, right_y = [], []
#         for line in right_lines:
#             x1, y1, x2, y2 = line[0]
#             right_x += [x1, x2]
#             right_y += [y1, y2]
#         if len(right_x) > 0 and len(right_y) > 0:
#             right_fit = np.polyfit(right_x, right_y, 1)
#             right_slope, right_intercept = right_fit

#             # Extrapolate the right lane line to the bottom and top of the lane
#             right_y1 = img.shape[0]
#             right_x1 = int((right_y1 - right_intercept) / right_slope)
#             right_y2 = int(img.shape[0] * 0.6)
#             right_x2 = int((right_y2 - right_intercept) / right_slope)
#             cv2.line(line_img, (right_x1, right_y1), (right_x2, right_y2), (0, 255, 0), thickness=10)

#     # Add the line image to the original image with a weight of 0.8
#     img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
#     return img



# # def draw_lines(img, lines):
# #     img = np.copy(img)
# #     line_img = np.zeros_like(img)
# #     if lines is not None:
# #         left_points = []
# #         right_points = []
# #         for line in lines:
# #             for x1, y1, x2, y2 in line:
# #                 if x1 < img.shape[1] / 2 and x2 < img.shape[1] / 2:
# #                     left_points.append((x1, y1))
# #                     left_points.append((x2, y2))
# #                     cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=5) # Left lane: Red
# #                 else:
# #                     right_points.append((x1, y1))
# #                     right_points.append((x2, y2))
# #                     cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), thickness=5) # Right lane: Blue
# #         if left_points and right_points:
# #             left_points = sorted(left_points, key=lambda x: x[1], reverse=True)
# #             right_points = sorted(right_points, key=lambda x: x[1], reverse=True)
# #             left_points = np.array(left_points)
# #             right_points = np.array(right_points)
# #             points = np.concatenate((left_points, np.flip(right_points, axis=0)), axis=0)
# #             cv2.fillPoly(line_img, [points], (0, 255, 0)) # Green polygon: Fill the space between the lanes
# #     img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
# #     return img


# # Define the function to process each frame from the webcam
# def process_image(img):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to the image
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply Canny edge detection to the image
#     edges = cv2.Canny(blur, 50, 150)
    
#     # Define the region of interest for the lane detection
#     height, width = img.shape[:2]
#     roi_vertices = [(0, height), (width/2, height/2), (width, height)]
#     roi = region_of_interest(edges, np.array([roi_vertices], np.int32))
    
#     # Apply Hough Transform to detect the lane lines
#     lines = cv2.HoughLinesP(roi, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)
    
#     # Draw the detected lane lines on the image
#     img_with_lines = draw_lines(img, lines)
    
#     return img_with_lines

# # Open the webcam
# cap = cv2.VideoCapture('project_video.mp4')

# while cap.isOpened():
#     # Read a frame from the webcam
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Process the frame
#     processed_frame = process_image(frame)
    
#     # Show the processed frame
#     cv2.imshow('Lane Detection', processed_frame)
    
#     # Exit if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()









# import cv2
# import numpy as np

# # Define the region of interest (ROI) for lane detection
# def region_of_interest(img, vertices):
#     mask = np.zeros_like(img)
#     match_mask_color = 255
#     cv2.fillPoly(mask, vertices, match_mask_color)
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image

# # Define the function to draw the detected lane lines
# def draw_lines(img, lines):
#     img = np.copy(img)
#     line_img = np.zeros_like(img)
#     if lines is not None:
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
#     img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
#     return img

# # Define the function to process each frame from the webcam
# def process_image(img):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to the image
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply Canny edge detection to the image
#     edges = cv2.Canny(blur, 50, 150)
    
#     # Define the region of interest for the lane detection
#     height, width = img.shape[:2]
#     roi_vertices = [(0, height), (width/2, height/2), (width, height)]
#     roi = region_of_interest(edges, np.array([roi_vertices], np.int32))
    
#     # Apply Hough Transform to detect the lane lines
#     lines = cv2.HoughLinesP(roi, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)
    
#     # Draw the detected lane lines on the image
#     img_with_lines = draw_lines(img, lines)
    
#     return img_with_lines

# # Open the webcam
# cap = cv2.VideoCapture('project_video.mp4')

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
    
#     # Process the frame
#     processed_frame = process_image(frame)
    
#     # Show the processed frame
#     cv2.imshow('Lane Detection', processed_frame)
    
#     # Exit if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Define the region of interest (ROI) for lane detection
# def region_of_interest(img, vertices):
#     mask = np.zeros_like(img)
#     match_mask_color = 255
#     cv2.fillPoly(mask, vertices, match_mask_color)
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image

# # Define the function to draw the detected lane lines
# def draw_lines(img, lines):
#     img = np.copy(img)
#     line_img = np.zeros_like(img)
#     if lines is not None:
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
#     img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
#     return img

# # Define the function to process each frame from the webcam
# def process_image(img):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to the image
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply Canny edge detection to the image
#     edges = cv2.Canny(blur, 50, 150)
    
#     # Define the region of interest for the lane detection
#     height, width = img.shape[:2]
#     roi_vertices = [(0, height), (width/2, height/2), (width, height)]
#     roi = region_of_interest(edges, np.array([roi_vertices], np.int32))
    
#     # Apply Hough Transform to detect the lane lines
#     lines = cv2.HoughLinesP(roi, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)
    
#     # Draw the detected lane lines on the image
#     img_with_lines = draw_lines(img, lines)
    
#     return img_with_lines

# # Open the webcam
# cap = cv2.VideoCapture('project_video.mp4')

# while cap.isOpened():
#     # Read a frame from the webcam
#     ret, frame = cap.read()
    
#     # Process the frame
#     processed_frame = process_image(frame)
    
#     # Show the processed frame
#     cv2.imshow('Lane Detection', processed_frame)
    
#     # Exit if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()






# rough-----------------------------------------


# import cv2
# import numpy as np

# # Define the region of interest (ROI) for lane detection
# def region_of_interest(img, vertices):
#     mask = np.zeros_like(img)
#     match_mask_color = 255
#     cv2.fillPoly(mask, vertices, match_mask_color)
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image


# def draw_lines(img, lines):
#     img = np.copy(img)
#     line_img = np.zeros_like(img)
#     if lines is not None:
#         # Separate the lines into left and right lane lines based on their slopes
#         left_lines, right_lines = [], []
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             if x2 - x1 == 0:  # Avoid dividing by zero
#                 continue
#             slope = (y2 - y1) / (x2 - x1)
#             if slope < 0:
#                 left_lines.append(line)
#             elif slope > 0:
#                 right_lines.append(line)

#         # Fit a line to the left lane lines using linear regression
#         left_x, left_y = [], []
#         for line in left_lines:
#             x1, y1, x2, y2 = line[0]
#             left_x += [x1, x2]
#             left_y += [y1, y2]
#         if len(left_x) > 0 and len(left_y) > 0:
#             left_fit = np.polyfit(left_x, left_y, 1)
#             left_slope, left_intercept = left_fit

#             # Extrapolate the left lane line to the bottom and top of the lane
#             left_y1 = img.shape[0]
#             left_x1 = int((left_y1 - left_intercept) / left_slope)
#             left_y2 = int(img.shape[0] * 0.6)
#             left_x2 = int((left_y2 - left_intercept) / left_slope)

#             # Draw the left lane line
#             cv2.line(line_img, (left_x1, left_y1), (left_x2, left_y2), (0, 255, 0), thickness=10)

#         # Fit a line to the right lane lines using linear regression
#         right_x, right_y = [], []
#         for line in right_lines:
#             x1, y1, x2, y2 = line[0]
#             right_x += [x1, x2]
#             right_y += [y1, y2]
#         if len(right_x) > 0 and len(right_y) > 0:
#             right_fit = np.polyfit(right_x, right_y, 1)
#             right_slope, right_intercept = right_fit

#             # Extrapolate the right lane line to the bottom and top of the lane
#             right_y1 = img.shape[0]
#             right_x1 = int((right_y1 - right_intercept) / right_slope)
#             right_y2 = int(img.shape[0] * 0.6)
#             right_x2 = int((right_y2 - right_intercept) / right_slope)

#             # Draw the right lane line
#             cv2.line(line_img, (right_x1, right_y1), (right_x2, right_y2), (0, 255, 0), thickness=10)

#         # Draw the green box between the two lane lines
#         pts_left = np.array([[left_x1, left_y1], [left_x2, left_y2], [right_x2, right_y2], [right_x1, right_y1]], np.int32)
#         cv2.fillPoly(line_img, [pts_left], (0, 255, 0))

#     # Add the line image to the original image with a weight of 0.

#     img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
#     return img


# # Define the function to process each frame from the webcam
# def process_image(img):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to the image
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply Canny edge detection to the image
#     edges = cv2.Canny(blur, 50, 150)
    
#     # Define the region of interest for the lane detection
#     height, width = img.shape[:2]
#     roi_vertices = [(0, height), (width/2, height/2), (width, height)]
#     roi = region_of_interest(edges, np.array([roi_vertices], np.int32))
    
#     # Apply Hough Transform to detect the lane lines
#     lines = cv2.HoughLinesP(roi, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)
    
#     # Draw the detected lane lines on the image
#     img_with_lines = draw_lines(img, lines)
    
#     return img_with_lines

# # Open the webcam
# cap = cv2.VideoCapture('project_video.mp4')

# while cap.isOpened():
#     # Read a frame from the webcam
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Process the frame
#     processed_frame = process_image(frame)
    
#     # Show the processed frame
#     cv2.imshow('Lane Detection', processed_frame)
    
#     # Exit if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()















