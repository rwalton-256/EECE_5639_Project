import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

P1 = np.array([
    [3.18019960e+03, 0.00000000e+00, 1.92428143e+03, 0.00000000e+00],
    [0.00000000e+00, 3.18131959e+03, 1.07018431e+03, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]
])

P2 = np.array([
    [3.71448717e+03, -5.85014142e+01, 3.25311697e+02, 4.33265470e+03],
    [5.12747211e+02, 3.11648947e+03, 1.19801542e+03, 4.39193783e+02],
    [4.38567875e-01, -6.45848787e-02, 8.96374371e-01, 4.25460086e-01]
])

cam1_track = []
with open("cam1_track_test3.json", 'r') as f:
    data = json.load(f)
    cam1_track = np.array(list(data.values()))

cam2_track = []
with open("cam2_track_test3.json", 'r') as f:
    data = json.load(f)
    cam2_track = np.array(list(data.values()))

# window length for the smoothing filter
window_length = 51
# order of the polynomial for the smoothing filter
poly_order = 3

# Apply Savitzky-Golay smoothing
list_of_u1_smoothed = savgol_filter(cam1_track[:, 0], window_length, poly_order)
list_of_v1_smoothed = savgol_filter(cam1_track[:, 1], window_length, poly_order)
list_of_u2_smoothed = savgol_filter(cam2_track[:, 0], window_length, poly_order)
list_of_v2_smoothed = savgol_filter(cam2_track[:, 1], window_length, poly_order)

points_3d = []
for u1, v1, u2, v2 in zip(list_of_u1_smoothed, list_of_v1_smoothed, list_of_u2_smoothed, list_of_v2_smoothed):
    points1 = np.array([[u1, v1]]).T
    points2 = np.array([[u2, v2]]).T
    homogeneous_points_3d = cv2.triangulatePoints(P1, P2, points1, points2)
    points_3d.append((homogeneous_points_3d[:3] / homogeneous_points_3d[3]).flatten())

points_3d = np.array(points_3d)

delta_t = 1 / 30
velocities = np.diff(points_3d, axis=0) / delta_t

# Convert velocities to meters per second
# velocity_x_mps = velocities[:, 0]
# velocity_y_mps = velocities[:, 1]
# velocity_z_mps = velocities[:, 2]

velocity_x_mps = savgol_filter(velocities[:, 0], window_length, poly_order)
velocity_y_mps = savgol_filter(velocities[:, 1], window_length, poly_order)
velocity_z_mps = savgol_filter(velocities[:, 2], window_length, poly_order)

# Compute magnitude of overall velocity in m/s
magnitude_speeds_mps = np.linalg.norm(velocities, axis=1)

magnitude_speeds_mps = savgol_filter(magnitude_speeds_mps, window_length, poly_order)

# Generate the time array for the plot
time = np.arange(len(magnitude_speeds_mps)) * delta_t

# Generate the plot
plt.figure()
plt.plot(time, velocity_x_mps, label='x-component velocity')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
# plt.xlim([0, 15])
# plt.ylim([0, 12])
plt.show()

plt.figure()
plt.plot(time, velocity_y_mps, label='y-component velocity')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
# plt.xlim([0, 15])
# plt.ylim([0, 12])
plt.show()

plt.figure()
plt.plot(time, velocity_z_mps, label='z-component velocity')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
# plt.xlim([0, 15])
# plt.ylim([0, 12])
plt.show()

plt.figure()
plt.plot(time, magnitude_speeds_mps, label='Velocity Magnitude')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
# plt.xlim([0, 15])
# plt.ylim([0, 12])
plt.show()