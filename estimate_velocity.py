import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Conversion factor from feet to meters
ft_to_m = 0.3048

P1 = np.array([
    [3180.19960, 0, 1924.28143, 0],
    [0, 3181.31959, 1070.18431, 0],
    [0, 0, 1, 0]
])

P2 = np.array([
    [3714.48717, -58.5014142, 325.311697, 14299.4368],
    [512.747211, 3116.48947, 1198.01542, 1449.5094],
    [0.438567875, -0.0645848787, 0.896374371, 1.40418289]
])

cam1_track = []
with open("cam1_track_test1.json", 'r') as f:
    data = json.load(f)
    cam1_track = np.array(list(data.values()))

cam2_track = []
with open("cam2_track_test1.json", 'r') as f:
    data = json.load(f)
    cam2_track = np.array(list(data.values()))

list_of_u1, list_of_v1 = cam1_track[:, 0], cam1_track[:, 1]
list_of_u2, list_of_v2 = cam2_track[:, 0], cam2_track[:, 1]

points_3d = []
for u1, v1, u2, v2 in zip(list_of_u1, list_of_v1, list_of_u2, list_of_v2):
    points1 = np.array([[u1, v1]]).T
    points2 = np.array([[u2, v2]]).T
    homogeneous_points_3d = cv2.triangulatePoints(P1, P2, points1, points2)
    points_3d.append((homogeneous_points_3d[:3] / homogeneous_points_3d[3]).flatten())

points_3d = np.array(points_3d)

delta_t = 1 / 30
velocities = np.diff(points_3d, axis=0) / delta_t

# Remove the z-component, y-component, or x-component of the velocity
velocities[:, 2] = 0
# velocities[:, 1] = 0
# velocities[:, 0] = 0

# Convert velocities to magnitude and then to meters per second
speeds_mps = np.linalg.norm(velocities, axis=1) * ft_to_m

# Apply Savitzky-Golay smoothing
window_length = 51  # Window length should be odd and >= the order of the polynomial.
poly_order = 3  # Order of polynomial to fit to the data for the smoothing filter
smoothed_speeds_mps = savgol_filter(speeds_mps, window_length, poly_order)

# Generate the time array for the plot
time = np.arange(len(smoothed_speeds_mps)) * delta_t

# Output the speeds
print("Smoothed One-Dimensional Speeds (m/s):")
print(smoothed_speeds_mps)

# Generate the plot
plt.figure()
plt.plot(time, smoothed_speeds_mps)
plt.xlabel('Time (s)')
plt.ylabel('Smoothed Speed (m/s)')
# plt.xlim([0, 15])
# plt.ylim([0, 3])

plt.show()
