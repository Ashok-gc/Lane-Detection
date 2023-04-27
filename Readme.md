# Lane Detection
This code performs lane detection on a video stream using the OpenCV library in Python. The code detects the lanes on the road and draws them on the original image to highlight them.

# Getting Started
# Prerequisites
1. Python 3.x
2. OpenCV 4.x

# Installation
1. Install Python 3.x from Python official website
2. Install OpenCV library via pip by running the following command in the terminal: `pip install opencv-python`

# Usage
1. Save the input video as "lanes_clip.mp4" in the same directory as the code.
2. Open the terminal and navigate to the directory where the code is saved.
3. Run the code by executing the following command: 
through video `python lane_detection.py` 
or
through camera: `python lane_detecction_camera.py`
or
final: `final.py`
4. The output video with the detected lanes will be displayed on the screen.
5. Press 'q' on the keyboard to quit the program.

# Algorithm
The code performs the following steps to detect lanes on the road:

1. Load the input video and apply a Gaussian blur to it.
2. Convert the blurred image to grayscale and apply Canny edge detection to detect edges.
3. Define the region of interest where the lane lines are expected to be present.
4. Apply a mask to the edge-detected image to keep only the region of interest.
5. Use Hough line detection to detect lines in the region of interest.
6. Classify the detected lines into left and right lane lines based on their slopes.
7. Average the slope and y-intercept of the detected lines for both left and right lanes separately.
8. Calculate the coordinates of the left and right lane lines based on the averaged slope and y-intercept.
9. Draw the left and right lane lines on a blank image and combine it with the original image to get the final output.


