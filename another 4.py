import cv2
import numpy as np

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150)
    return canny

def detect_lane_lines(frame):
    lines = cv2.HoughLinesP(frame, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    left_lane = []
    right_lane = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_lane.append((slope, intercept))
        else:
            right_lane.append((slope, intercept))

    left_lane_avg = np.average(left_lane, axis=0)
    right_lane_avg = np.average(right_lane, axis=0)

    return left_lane_avg, right_lane_avg

def create_line_coordinates(frame, parameters):
    slope, intercept = parameters
    y1 = frame.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])

def track_lanes(frame, left_lane, right_lane):
    left_line = create_line_coordinates(frame, left_lane)
    right_line = create_line_coordinates(frame, right_lane)

    return np.array([left_line, right_line])

def color_lane_boundaries(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    green_overlay = np.zeros_like(frame)
    polygon = np.array([[
        [lines[0][0], lines[0][1]],
        [lines[0][2], lines[0][3]],
        [lines[1][2], lines[1][3]],
        [lines[1][0], lines[1][1]]
    ]], dtype=np.int32)
    cv2.fillPoly(green_overlay, polygon, (0, 255, 0))
    combined_image = cv2.addWeighted(frame, 0.8, green_overlay, 0.2, 0)
    return cv2.addWeighted(combined_image, 0.8, line_image, 1, 0)


def lane_departure_warning(lines, frame_width):
    left_lane, right_lane = lines
    left_bottom_x = left_lane[0]
    right_bottom_x = right_lane[0]
    
    lane_center = (right_bottom_x - left_bottom_x) // 2 + left_bottom_x
    vehicle_position = frame_width // 2
    threshold = frame_width * 0.05

    if abs(lane_center - vehicle_position) > threshold:
        return "WARNING: Lane Departure"
    else:
        return ""

def main():
    cap = cv2.VideoCapture('project_video.mp4')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame)
        left_lane, right_lane = detect_lane_lines(preprocessed_frame)
        tracked_lanes = track_lanes(frame, left_lane, right_lane)
        colored_frame = color_lane_boundaries(frame, tracked_lanes)
        
        lane_warning = lane_departure_warning(tracked_lanes, frame_width)
        if lane_warning:
            cv2.putText(colored_frame, lane_warning, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Lane Detection', colored_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
