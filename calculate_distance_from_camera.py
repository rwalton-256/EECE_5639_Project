# Importing numpy package to perform array operations
import numpy as np

# Define height and calculate target difference
h = 3200  # Height of camera
c = 1720  # Target height
t = h - c  # The difference in height and target

# Create a numpy array for camera matrix
camera_matrix = np.array(
    [
        [4.57460952e+03, 0.00000000e+00, 1.99943325e+03],
        [0.00000000e+00, 9.60875719e+03, 1.12261263e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ]
)

# Set the focal length of the camera
focal_length = 24

# Calculate the sensor's pixel width and height
sensor_pixel_width = 2 * camera_matrix[0][2]
# Double the number of horizontal pixel elements in the camera matrix 
sensor_pixel_height = 2 * camera_matrix[1][2]
# Double the number of vertical pixel elements in the camera matrix 

# Get the focal lengths in pixel dimensions for x and y directions 
fx = camera_matrix[0][0]
fy = camera_matrix[1][1]

# Set the image-wide pixels
img_width_px = 4000
img_height_px = 2250

# Define sensor dimensions in mm (not used atm)
sensor_width_mm = 6.16
sensor_height_mm = 4.62

# calculate pixel per mm based on focal length
m = sensor_pixel_height / focal_length
s = abs(1577 - 668)  # absolute pixel height of target in the image

# Calculate updated target height on sensor and height difference on the sensor based on pixel count
c_prime = s / m
t_prime = t * (c_prime / c)

# Calculate the distance from camera to object in picture plane using Pythagorean theorem
a_prime = np.sqrt(focal_length ** 2 + (c_prime + t_prime) ** 2)
a = a_prime * (c / c_prime)

# calculate the final distance
d = np.sqrt(a ** 2 - (t + c) ** 2)

# Print distances in different units
print("Distance: ", d, "mm")  # Print the distance in millimeters
print("Distance: ", d / 10, "cm")  # Print the distance in centimeters

# The above code calculates the distance to an object based on image pixel data, camera settings and physical dimensions.
