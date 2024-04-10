import os
import glob
import cv2
import numpy as np


def calibrate_camera(directory_path, chessboard_size=(10, 7)):
    # Termination criteria for corner subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Prepare object points, like (0,0,0), (1,0,0), ..., (6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []
    imgpoints = []
    universe_size = None

    # Get a list of all image file paths in the directory
    image_paths = glob.glob(os.path.join(directory_path, "*.JPG"))

    # Create a directory to save the correspondence images
    new_dir_path = os.path.join(directory_path, "correspondence_images")
    os.makedirs(new_dir_path, exist_ok=True)

    for image_path in image_paths:
        # Read each image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        universe_size = gray.shape[::-1]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            # Add the object points and the found corners to the respective lists
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Show the correspondence by drawing the corners
            img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            # Save the correspondence image
            image_name = os.path.basename(image_path)  # get the name of the original file
            new_path = os.path.join(new_dir_path, image_name)  # create a new path in the correspondence_images directory
            cv2.imwrite(new_path, img)

    # Calibrate the camera using all the object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, universe_size, None, None)

    if ret:
        print("Camera Matrix:")
        print(mtx)

        print("Distortion Matrix:")
        print(dist)
    else:
        print("Camera calibration unsuccessful.")


if __name__ == "__main__":
    directory_path = 'drone_calibration'
    calibrate_camera(directory_path)
