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

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap):
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)


# def draw_lines(image, lines, left_color, right_color, thickness):
#     if lines is None:
#         return

#     left_lines = []
#     right_lines = []
#     min_slope = 0.5
#     max_slope = 2.0

#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             if x2 == x1:
#                 continue
#             slope = (y2 - y1) / (x2 - x1)
#             intercept = y1 - slope * x1
#             if min_slope < abs(slope) < max_slope:
#                 if slope < 0:
#                     left_lines.append((slope, intercept))
#                 else:
#                     right_lines.append((slope, intercept))

#     y1 = int(image.shape[0] * 0.6)
#     y2 = image.shape[0]
#     y = np.array([y1, y2])

#     def draw_lane_lines(lines, color):
#         if lines:
#             slope, intercept = np.mean(lines, axis=0)
#             x1, x2 = (y - intercept) / slope
#             x1, x2 = int(x1), int(x2)
#             cv2.line(image, (x1, y1), (x2, y2), color, thickness)

#     draw_lane_lines(left_lines, left_color)
#     draw_lane_lines(right_lines, right_color)
def draw_lines(image, lines, left_color, right_color, thickness):
    if lines is None:
        return None, None

    left_lines = []
    right_lines = []
    min_slope = 0.5
    max_slope = 2.0

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if min_slope < abs(slope) < max_slope:
                if slope < 0:
                    left_lines.append((slope, intercept))
                else:
                    right_lines.append((slope, intercept))

    y1 = int(image.shape[0] * 0.6)
    y2 = image.shape[0]
    y = np.array([y1, y2])

    def draw_lane_lines(lines, color):
        if lines:
            slope, intercept = np.mean(lines, axis=0)
            x1, x2 = (y - intercept) / slope
            x1, x2 = int(x1), int(x2)
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
            return [(x1, y1), (x2, y2)]
        return None

    left_points = draw_lane_lines(left_lines, left_color)
    right_points = draw_lane_lines(right_lines, right_color)

    return left_points, right_points


# def process_image(image, parameters):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))
#     white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
#     color_mask = cv2.bitwise_or(yellow_mask, white_mask)
    
#     gray = grayscale(image)
#     blurred_gray = gaussian_blur(gray, parameters["gaussian_kernel"])
#     edges = canny(blurred_gray, parameters["canny_low"], parameters["canny_high"])
#     color_edges = cv2.bitwise_and(edges, color_mask)
#     masked_edges = region_of_interest(color_edges, parameters["roi_vertices"])
#     lines = hough_lines(masked_edges, parameters["hough_rho"],
#         parameters["hough_theta"], parameters["hough_threshold"], parameters["min_line_length"], parameters["max_line_gap"])
#     line_image = np.copy(image)
#     draw_lines(line_image, lines, parameters["left_color"], parameters["right_color"], parameters["line_thickness"])


#     left_points, right_points = draw_lines(line_image, lines, parameters["left_color"], parameters["right_color"], parameters["line_thickness"])

#     if left_points is not None and right_points is not None:
#         cv2.fillPoly(line_image, [np.array([left_points[0], left_points[1], right_points[1], right_points[0]])], (0, 255, 0))

#     return line_image

def process_image(image, parameters):
    def color_filter(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define color ranges for white and yellow
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 25, 255])

        yellow_lower = np.array([20, 80, 80])
        yellow_upper = np.array([40, 255, 255])

        # Threshold the HSV image to get only white and yellow colors
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        # Combine the two masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Bitwise-AND the mask and the original image
        return cv2.bitwise_and(img, img, mask=combined_mask)

    filtered_image = color_filter(image)
    gray = grayscale(filtered_image)
    blurred = gaussian_blur(gray, parameters["gaussian_kernel"])
    edges = canny(blurred, parameters["canny_low"], parameters["canny_high"])
    masked_edges = region_of_interest(edges, parameters["roi_vertices"])
    lines = hough_lines(masked_edges, parameters["hough_rho"], parameters["hough_theta"], parameters["hough_threshold"], parameters["min_line_length"], parameters["max_line_gap"])
    line_image = np.copy(image)
    left_points, right_points = draw_lines(line_image, lines, parameters["left_color"], parameters["right_color"], parameters["line_thickness"])

    if left_points is not None and right_points is not None:
        cv2.fillPoly(line_image, [np.array([left_points[0], left_points[1], right_points[1], right_points[0]])], (0, 255, 0))

    return line_image


def main(input_video, output_video, parameters):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_image(frame, parameters)
        out.write(processed_frame)

        cv2.imshow('frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = 'clip.mp4'
    output_video = 'output_video.mp4'

    parameters = {
        "gaussian_kernel": 5,
        "canny_low": 50,
        "canny_high": 150,
        "roi_vertices": np.array([[(100, 540), (960, 540), (650, 350), (320, 350)]], dtype=np.int32),
        "hough_rho": 2,
        "hough_theta": np.pi / 180,
        "hough_threshold": 40,
        "min_line_length": 30,
        "max_line_gap": 20,
        "left_color": (255, 0, 0),
        "right_color": (0, 0, 255),
        "line_thickness": 8,
    }

    main(input_video, output_video, parameters)
