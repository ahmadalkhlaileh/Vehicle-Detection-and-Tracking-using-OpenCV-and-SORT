import cv2
import numpy as np
import os
from time import sleep
from sort import Sort

# Parameters
min_width = 70
min_height = 70
offset = 20
delay = 10
video_path = 'tracking_3.avi'
output_video_path = 'output_video.avi'
output_directory = 'vehicle_images'
annotation_directory = 'annotations'
classes = 1  # Adjust according to your dataset, assuming 1 class for vehicles

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Initialize background subtractor
subtraction = cv2.createBackgroundSubtractorMOG2()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize SORT tracker
tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

# Initialize dictionary to store tracked objects and their IDs
tracked_objects_dict = {}

# Create output directory for vehicle images and annotations
os.makedirs(output_directory, exist_ok=True)
os.makedirs(annotation_directory, exist_ok=True)

# Initialize variables for initialization phase
frame_count = 0
initialized = False

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    sleep(1 / delay)

    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    img_sub = subtraction.apply(blur)
    dilated = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Detect objects
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = [
        [cv2.boundingRect(contour)[0], cv2.boundingRect(contour)[1], cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2], cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3], 1]
        for contour in contours
        if cv2.boundingRect(contour)[2] >= min_width and cv2.boundingRect(contour)[3] >= min_height
    ]

    # Debugging: Print the shape of detected_objects
    print(f"Detected Objects: {detected_objects}")
    if detected_objects:
        print(f"Shape of detected_objects: {np.array(detected_objects).shape}")

    # Update tracker only after initialization phase
    if initialized:
        if detected_objects:
            tracked_objects = tracker.update(np.array(detected_objects))
        else:
            tracked_objects = []
    else:
        tracked_objects = []

    # Process tracked objects
    for obj in tracked_objects:
        bbox = obj[:4].astype(int)
        track_id = int(obj[4])
        
        # Ensure bounding box coordinates are within frame bounds
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(width - 1, bbox[2])
        bbox[3] = min(height - 1, bbox[3])

        # Calculate bounding box width and height
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        # Check if bounding box width and height are valid
        if bbox_width > 0 and bbox_height > 0:
            # Take a screenshot of the entire frame if a new object is detected
            if track_id not in tracked_objects_dict:
                tracked_objects_dict[track_id] = len(tracked_objects_dict) + 1
                screenshot_path = os.path.join(output_directory, f'screenshot_{tracked_objects_dict[track_id]}.jpg')
                cv2.imwrite(screenshot_path, frame)
                
                # Create corresponding annotation file
                annotation_path = os.path.join(annotation_directory, f'screenshot_{tracked_objects_dict[track_id]}.txt')
                with open(annotation_path, 'w') as f:
                    # Calculate normalized coordinates
                    x_center = (bbox[0] + bbox[2]) / 2 / width
                    y_center = (bbox[1] + bbox[3]) / 2 / height
                    w = bbox_width / width
                    h = bbox_height / height
                    f.write(f"0 {x_center} {y_center} {w} {h}\n")
                
            # Extract and save vehicle image (optional, if needed)
            vehicle_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if vehicle_image.size != 0:
                vehicle_image_path = os.path.join(output_directory, f'vehicle_{tracked_objects_dict[track_id]}.jpg')
                cv2.imwrite(vehicle_image_path, vehicle_image)
                
            # Draw rectangle and ID (if needed, can be commented out)
            # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # cv2.putText(frame, f'ID: {tracked_objects_dict[track_id]}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    cv2.imshow("Video Original", frame)
    if cv2.waitKey(1) == 27:
        break

    # Check if initialization phase is completed
    if not initialized and frame_count >= 10:
        initialized = True

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_video_path}")
print(f"Vehicle images and annotations saved to {output_directory} and {annotation_directory}")
