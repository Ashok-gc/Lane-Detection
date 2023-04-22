import cv2
import numpy as np

# Set up video capture
cap = cv2.VideoCapture('project_video.mp4')

# Define the region of interest
vertices = np.array([[0, 720], [640, 480], [800, 480], [1280, 720]], dtype=np.int32)

# Define the Hough transform parameters
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 30 # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50 # minimum number of pixels making up a line
max_line_gap = 100 # maximum gap in pixels between connectable line segments

# Define a function to perform lane detection on a single frame
def detect_lanes(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Mask the edges image to only show the region of interest
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [vertices], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Perform Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    
    # Draw the detected lines on the original frame
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Blend the original frame with the line image
    blended_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    
    return blended_image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Perform lane detection on the current frame
    lane_detection = detect_lanes(frame)
    
    # Display the resulting image
    cv2.imshow("Lane Detection", lane_detection)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
