import random
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# Initialise the two models
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n.pt')

# Load videos 1 and 2
video1_path = "vid1" + ".mp4"
cap1 = cv2.VideoCapture(video1_path)
fps1 = cap1.get(cv2.CAP_PROP_FPS)
image_width1, image_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

video2_path = "vid2" + ".mp4"
cap2 = cv2.VideoCapture(video2_path)
fps2 = cap2.get(cv2.CAP_PROP_FPS)
image_width2, image_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Writes the processed videos to these files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out1 = cv2.VideoWriter('output1.mp4', fourcc, fps1, (image_width1, image_height1))
out2 = cv2.VideoWriter('output2.mp4', fourcc, fps2, (image_width2, image_height2))

# An array to store the unique color for each identified target
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

# This stores the tracks for each detected target
track_history1 = defaultdict(lambda: [])
track_history2 = defaultdict(lambda: [])

# IMPORTANT: IF YOU ARE NOT RUNNING ON AN APPLE SILICON DEVICE, CHANGE MPS TO EITHER CPU OR NVIDIA, DEPENDING ON WHAT
# YOU HAVE
model1.to("mps")
model2.to("mps")

model1.share_memory(model2)

while True:

    # Read frames from both video streams
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Break the loop if either video stream is finished
    if not ret1 or not ret2:
        break

    # Process the first frame using the first model. IMPORTANT: IF YOU ARE NOT RUNNING ON AN APPLE SILICON DEVICE,
    # CHANGE MPS TO EITHER CPU OR NVIDIA, DEPENDING ON WHAT YOU HAVE
    results1 = model1.track(frame1, classes=[0], verbose=False, persist=True, tracker="bytetrack.yaml", device="mps")

    # Process the second frame using the second model. IMPORTANT: IF YOU ARE NOT RUNNING ON AN APPLE SILICON DEVICE,
    # CHANGE MPS TO EITHER CPU OR NVIDIA, DEPENDING ON WHAT YOU HAVE
    results2 = model2.track(frame2, classes=[0], verbose=False, persist=True, tracker="bytetrack.yaml", device="mps")

    # Plot the processed frames and the results of the models
    annotated_frame1 = results1[0].plot()
    annotated_frame2 = results2[0].plot()

    # Process all detections in video 1 to plot the trajectory of the target(s) on the frame
    if results1[0].boxes.target_id is not None:
        boxes = results1[0].boxes.xywh.cpu().numpy()
        track_ids = results1[0].boxes.target_id.int().cpu().numpy()

        for i in range(boxes.shape[0]):
            x, y, w, h = boxes[i]
            target_id = track_ids[i]
            color = colors[target_id]
            track_history1[target_id].append((float(x), float(y)))
            pts1 = np.array(track_history1[target_id], np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame1, [pts1], isClosed=True, color=color, thickness=3)

    # Process all detections in video 2 to plot the trajectory of the target(s) on the frame
    if results2[0].boxes.target_id is not None:
        boxes = results1[0].boxes.xywh.cpu().numpy()
        track_ids = results1[0].boxes.target_id.int().cpu().numpy()

        for i in range(boxes.shape[0]):
            x, y, w, h = boxes[i]
            target_id = track_ids[i]
            color = colors[target_id]
            track_history2[target_id].append((float(x), float(y)))

            pts2 = np.array(track_history2[target_id], np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame2, [pts2], isClosed=True, color=color, thickness=3)

    # Display the processed frames
    cv2.imshow('Video 1', annotated_frame1)
    cv2.imshow('Video 2', annotated_frame2)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video streams and close the windows
cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
