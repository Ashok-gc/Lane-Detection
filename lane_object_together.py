# # import cv2
# # import numpy as np
# # thres = 0.45

# # # Load class names
# # classNames = []
# # classFile = 'coco.names'
# # with open(classFile, 'rt') as f:
# #     classNames = f.read().rstrip('\n').split('\n')

# # # Load model
# # configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# # weightsPath = 'frozen_inference_graph.pb'
# # net = cv2.dnn_DetectionModel(weightsPath, configPath)
# # net.setInputSize(320,320)
# # net.setInputScale(1.0/127.5)
# # net.setInputMean((127.5, 127.5, 127.5))
# # net.setInputSwapRB(True)

# # # Load the video
# # cap = cv2.VideoCapture('clip.mp4')

# # # Define the region of interest
# # roi_vertices = [(0, 720), (1280, 720), (750, 460), (550, 460)]
# # def region_of_interest(img, vertices):
# #     mask = np.zeros_like(img)
# #     if len(img.shape) > 2:
# #         match_mask_color = (255,) * img.shape[2]
# #     else:
# #         match_mask_color = 255
# #     cv2.fillPoly(mask, [vertices], match_mask_color)
# #     masked_image = cv2.bitwise_and(img, mask)
# #     return masked_image

# # # Define the line drawing function
# # def draw_lines(img, lines):
# #     left_line_x = []
# #     left_line_y = []
# #     right_line_x = []
# #     right_line_y = []

# #     for line in lines:
# #         for x1, y1, x2, y2 in line:
# #             slope = (y2 - y1) / (x2 - x1)
# #             if slope < 0:
# #                 left_line_x.extend([x1, x2])
# #                 left_line_y.extend([y1, y2])
# #             else:
# #                 right_line_x.extend([x1, x2])
# #                 right_line_y.extend([y1, y2])

# #     if len(left_line_x) > 0:
# #         left_fit = np.polyfit(left_line_x, left_line_y, 1)
# #         left_line = np.poly1d(left_fit)
# #         left_start = (int(min(left_line_x)), int(left_line(min(left_line_x))))
# #         left_end = (int(max(left_line_x)), int(left_line(max(left_line_x))))
# #         cv2.line(img, left_start, left_end, (0, 255, 0), 10)


# #         left_line_coords = calculate_line_coordinates(img, left_fit)
# #         if left_line_coords is not None:
# #             left_line_coords = left_line_coords.reshape((-1, 2))
# #             left_line_coords = left_line_coords.astype(int)
# #             cv2.line(img, tuple(left_line_coords[0]), tuple(left_line_coords[1]), (0, 255, 0), 3)


# #     if len(right_line_x) > 0:
# #         right_fit = np.polyfit(right_line_x, right_line_y, 1)
# #         right_line = np.poly1d(right_fit)
# #         right_start = (int(min(right_line_x)), int(right_line(min(right_line_x))))
# #         right_end = (int(max(right_line_x)), int(right_line(max(right_line_x))))
# #         cv2.line(img, right_start, right_end, (0, 255, 0), 10)
    


# # def calculate_line_coordinates(img, parameters):
# #     if parameters is None or isinstance(parameters, np.float64):
# #         return None
# #     elif len(parameters) != 2:
# #         return None
# #     else:
# #         slope, y_intercept = parameters
# #         y1 = img.shape[0]
# #         y2 = int(y1 * 0.6)
# #         x1 = int((y1 - y_intercept) / slope)
# #         x2 = int((y2 - y_intercept) / slope)
# #         return np.array([x1, y1, x2, y2])

# # # Process each frame
# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     # Object detection
# #     classIds, confs, bbox = net.detect(frame, confThreshold=thres)
# #     if len(classIds) != 0:
# #         for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
# #             cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
# #             cv2.putText(frame, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
# #                         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

# # #     # Lane detection
# # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # #     edges = cv2.Canny(gray, 100, 200)
# # #     roi = region_of_interest(edges, np.array([roi_vertices], np.int32))
# # #     lines = cv2.HoughLinesP(roi, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)
# # #     draw_lines(frame, lines)

# # #     # Display the frame
# # #     cv2.imshow('Frame', frame)
# # #     if cv2.waitKey(1) == ord('q'):
# # #         break

# # # # Release the video
# # # cap.release()
# # # cv2.destroyAllWindows()

# # # Lane detection
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     edges = cv2.Canny(gray, 100, 200)
# #     roi = region_of_interest(edges, np.array([roi_vertices], np.int32))
# #     lines = cv2.HoughLinesP(roi, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)
    
# #     # Check if any lines are detected
# #     if lines is not None:
# #         left_line, right_line = None, None
# #         for line in lines:
# #             x1, y1, x2, y2 = line[0]
# #             slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
# #             line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
# #             if slope < 0:  # Left lane
# #                 if left_line is None or line_length > left_line[1]:
# #                     left_line = ((x1, y1, x2, y2), line_length)
# #             else:  # Right lane
# #                 if right_line is None or line_length > right_line[1]:
# #                     right_line = ((x1, y1, x2, y2), line_length)
# #         # Calculate and draw the lines
# #         left_line_coords = calculate_line_coordinates(frame, np.polyfit((left_line[0][0], left_line[0][2]),
# #                                                                        (left_line[0][1], left_line[0][3]), 1))
# #         right_line_coords = calculate_line_coordinates(frame, np.polyfit((right_line[0][0], right_line[0][2]),
# #                                                                          (right_line[0][1], right_line[0][3]), 1))
# #         if left_line_coords is not None:
# #             cv2.line(frame, (left_line_coords[0], left_line_coords[1]),
# #                      (left_line_coords[2], left_line_coords[3]), (0, 255, 0), 5)
# #         if right_line_coords is not None:
# #             cv2.line(frame, (right_line_coords[0], right_line_coords[1]),
# #                      (right_line_coords[2], right_line_coords[3]), (0, 255, 0), 5)

# #     # Display the frame
# #     cv2.imshow('Frame', frame)
# #     if cv2.waitKey(1) == ord('q'):
# #         break

# # # Release the video
# # cap.release()
# # cv2.destroyAllWindows()

# import cv2

# # Set threshold to detect objects
# thres = 0.45
# cap = cv2.VideoCapture('clip.mp4')
# # Load class names
# classNames = []
# classFile = 'coco.names'
# with open(classFile, 'rt') as f:
#     classNames = f.read().rstrip('\n').split('\n')

# # Load model
# configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# weightsPath = 'frozen_inference_graph.pb'
# net = cv2.dnn_DetectionModel(weightsPath, configPath)
# net.setInputSize(320,320)
# net.setInputScale(1.0/127.5)
# net.setInputMean((127.5, 127.5, 127.5))
# net.setInputSwapRB(True)

# # Load lane detection model
# lane_detection_model = cv2.createLineSegmentDetector()

# # For video clip or real time
# cap = cv2.VideoCapture('clip.mp4')
# cap.set(3, 1280)
# cap.set(4,720)
# cap.set(10,70)

# while True:
#     success, img = cap.read()

#     # Lane detection
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
#     lines = lane_detection_model.detect(thresh)
#     lane_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
#     lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2GRAY)
#     lane_img.fill(0)
#     for line in lines[0]:
#         x1, y1, x2, y2 = map(int, line[0])
#         cv2.line(lane_img, (x1, y1), (x2, y2), (255, 255, 255), 5)

#     # Object detection
#     classIds, confs, bbox = net.detect(img, confThreshold=thres)

#     # Draw bounding box and label for each object detected
#     if len(classIds) != 0:
#         for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
#             cv2.rectangle(img, box, color=(0,255,0),thickness=3)
#             cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
#                 cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
#             cv2.putText(img, str(round(confidence*100,2)), (box[0] + 200, box[1] + 30),
#                 cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

#     # Combine lane and object detection images
#     lane_img_bgr = cv2.cvtColor(lane_img, cv2.COLOR_GRAY2BGR)
#     combined_img = cv2.bitwise_and(lane_img_bgr, img)

#     # Display output image or video
#     cv2.imshow("Output",combined_img)

#     # Exit on 's' key press
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         break

# cap.release()
# cv2.destroyAllWindows()

