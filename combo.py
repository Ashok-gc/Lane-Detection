import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained object detection model
model = tf.keras.models.load_model("object_detection_model.h5")

# Define the function to detect objects
def detect_objects(image):
    # Preprocess the image for input to the object detection model
    resized_image = cv2.resize(image, (224, 224))
    input_image = np.expand_dims(resized_image, axis=0)
    input_image = input_image / 255.0

    # Run the object detection model on the image
    predictions = model.predict(input_image)

    # Get the class IDs and scores for the top 5 predictions
    top_ids = np.argsort(predictions[0])[-5:]
    top_scores = predictions[0][top_ids]

    # Get the class labels
    class_labels = ["person", "car", "truck", "motorcycle", "bus"]

    # Create a list of detected objects
    objects = []
    for i in range(len(top_ids)):
        if top_scores[i] > 0.5:
            object = {}
            object["label"] = class_labels[top_ids[i]]
            object["score"] = top_scores[i]
            objects.append(object)

    return objects

# Define the function to calculate the steering angle
def calculate_steering_angle(lane_lines, image_width):
    # Calculate the x-coordinate of the vanishing point
    left_line = lane_lines[0]
    right_line = lane_lines[1]
    x_left = (left_line[1] - image_width / 2) * (right_line[3] - right_line[1]) / (right_line[2] - right_line[0]) + left_line[0]
    x_right = (right_line[1] - image_width / 2) * (left_line[3] - left_line[1]) / (left_line[2] - left_line[0]) + right_line[0]
    x_vanishing = (x_left + x_right) / 2

    # Calculate the steering angle
    steering_angle = np.arctan((x_vanishing - image_width / 2) / (image_width / 2))

    return steering_angle

# Define the function to detect lanes and objects and calculate the steering angle
def process_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define a region of interest
    mask = np.zeros_like(edges)
    height, width = frame.shape[:2]
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Apply Hough transform to detect lane lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=300)

    # Draw the detected lane lines on the image
    lane_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Calculate the slope and intercept of the line
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Save the line's parameters
        lane_lines.append((slope, intercept))

    # Split the lane lines into left and right lines
    left_lines = []
    right_lines = []
    for slope, intercept in lane_lines:
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    # Calculate the average slope and intercept of the left lane line
    if len(left_lines) > 0:
        left_slope, left_intercept = np.mean(left_lines, axis=0)
    else:
        left_slope, left_intercept = None, None

    # Calculate the average slope and intercept of the right lane line
    if len(right_lines) > 0:
        right_slope, right_intercept = np.mean(right_lines, axis=0)
    else:
        right_slope, right_intercept = None, None

    # Draw the left and right lane lines on the image
    if left_slope and left_intercept:
        y1 = frame.shape[0]
        y2 = int(y1 / 2)
        x1 = int((y1 - left_intercept) / left_slope)
        x2 = int((y2 - left_intercept) / left_slope)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if right_slope and right_intercept:
        y1 = frame.shape[0]
        y2 = int(y1 / 2)
        x1 = int((y1 - right_intercept) / right_slope)
        x2 = int((y2 - right_intercept) / right_slope)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Detect objects in the image
    objects = detect_objects(frame)

    # Draw the detected objects on the image
    for object in objects:
        label = object["label"]
        score = object["score"]
        cv2.putText(frame, f"{label}: {score:.2f}", (10, int(30 * (label_index + 1))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate the steering angle
    lane_lines = [(left_slope, left_intercept), (right_slope, right_intercept)]
    steering_angle = calculate_steering_angle(lane_lines, frame.shape[1])

    # Draw the steering angle on the image
    cv2.putText(frame, f"Steering angle: {steering_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Apply the steering angle to control the vehicle
    control_vehicle(steering_angle)

    return frame
