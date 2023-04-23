# import cv2
# import numpy as np

# def grayscale(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# def canny(image, low_threshold, high_threshold):
#     return cv2.Canny(image, low_threshold, high_threshold)

# def region_of_interest(image, vertices):
#     mask = np.zeros_like(image)
#     cv2.fillPoly(mask, vertices, 255)
#     masked_image = cv2.bitwise_and(image, mask)
#     return masked_image

# def hough_lines(image, rho, theta, threshold, min_line_length, max_line_gap):
#     lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
#     return lines

# def draw_lines(image, lines, color=(0, 0, 255), thickness=2):
#     left_line_x = []
#     left_line_y = []
#     right_line_x = []
#     right_line_y = []

#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             slope = (y2 - y1) / (x2 - x1)
#             if slope < 0:
#                 left_line_x.extend([x1, x2])
#                 left_line_y.extend([y1, y2])
#             else:
#                 right_line_x.extend([x1, x2])
#                 right_line_y.extend([y1, y2])

#     if left_line_x and left_line_y:
#         left_fit = np.polyfit(left_line_y, left_line_x, 1)
#         left_line_fn = np.poly1d(left_fit)
#         cv2.line(image, (int(left_line_fn(image.shape[0])), image.shape[0]), (int(left_line_fn(image.shape[0] * 0.6)), int(image.shape[0] * 0.6)), color, thickness)

#     if right_line_x and right_line_y:
#         right_fit = np.polyfit(right_line_y, right_line_x, 1)
#         right_line_fn = np.poly1d(right_fit)
#         cv2.line(image, (int(right_line_fn(image.shape[0])), image.shape[0]), (int(right_line_fn(image.shape[0] * 0.6)), int(image.shape[0] * 0.6)), color, thickness)

# def process_image(image, parameters):
#     gray = grayscale(image)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = canny(blurred, parameters["canny_low"], parameters["canny_high"])
#     masked_edges = region_of_interest(edges, parameters["roi_vertices"])
#     lines = hough_lines(masked_edges, parameters["hough_rho"], parameters["hough_theta"], parameters["hough_threshold"], parameters["min_line_length"], parameters["max_line_gap"])
#     line_image = np.zeros_like(image)
#     draw_lines(line_image, lines, parameters["line_color"], parameters["line_thickness"])
#     result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
#     return result

# def main(input_video, output_video, parameters):
#     cap = cv2.VideoCapture(input_video)
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video, fourcc, 30.0, (640, 480))

#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret:
#             result = process_image(frame, parameters)
#             out.write(result)
#             cv2.imshow('Lane Detection', result)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     parameters = {
#         "canny_low": 50,
#         "canny_high": 150,
#         "roi_vertices": np.array([[(0, 480), (300, 300), (340, 300), (640, 480)]], dtype=np.int32),
#         "hough_rho": 1,
#         "hough_theta": np.pi/180,
#         "hough_threshold": 15,
#         "min_line_length": 40,
#         "max_line_gap": 20,
#         "line_color": (0, 0, 255),
#         "line_thickness": 5
#     }

#     main('project_video.mp4', 'output_video.avi', parameters)

import cv2
import numpy as np

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_lines(image, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

def draw_lines(image, lines, left_color, right_color, thickness):
    left_line_points = []
    right_line_points = []
    left_weights = []
    right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:
                left_line_points.append((slope, intercept))
                left_weights.append(length)
            else:
                right_line_points.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.dot(left_weights, left_line_points) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_line_points) / np.sum(right_weights) if len(right_weights) > 0 else None

    y1 = int(image.shape[0] * 0.6)
    y2 = image.shape[0]

    def calc_x(y, line):
        return int((y - line[1]) / line[0])

    if left_lane is not None:
        left_x1, left_x2 = calc_x(y1, left_lane), calc_x(y2, left_lane)
        cv2.line(image, (left_x1, y1), (left_x2, y2), left_color, thickness)

    if right_lane is not None:
        right_x1, right_x2 = calc_x(y1, right_lane), calc_x(y2, right_lane)
        cv2.line(image, (right_x1, y1), (right_x2, y2), right_color, thickness)

    if left_lane is not None and right_lane is not None:
        cv2.fillPoly(image, [np.array([(left_x1, y1), (left_x2, y2), (right_x2, y2), (right_x1, y1)])], (0, 255, 0))

def process_image(image, parameters):
    gray = grayscale(image)
    edges = canny(gray, parameters["canny_low"], parameters["canny_high"])
    masked_edges = region_of_interest(edges, parameters["roi_vertices"])
    lines = hough_lines(masked_edges, parameters["hough_rho"], parameters["hough_theta"], parameters["hough_threshold"], parameters["min_line_length"], parameters["max_line_gap"])
    line_image = np.copy(image)
    draw_lines(line_image, lines, parameters["left_color"], parameters["right_color"], parameters["line_thickness"])
    
    return line_image

def main(input_video, output_video, parameters):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (frame_width, frame_height))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            result = process_image(frame, parameters)
            out.write(result)
            cv2.imshow('Lane Detection', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parameters = {
        "canny_low": 50,
        "canny_high": 150,
        "roi_vertices": np.array([[(100, 540), (460, 325), (520, 325), (900, 540)]], dtype=np.int32),
        "hough_rho": 1,
        "hough_theta": np.pi/180,
        "hough_threshold": 15,
        "min_line_length": 40,
        "max_line_gap": 20,
        "left_color": (255, 0, 0),
        "right_color": (0, 0, 255),
        "line_thickness": 5
    }

    main('clip.mp4', 'output_video.avi', parameters)
