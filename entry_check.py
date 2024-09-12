import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model (move this outside the loop)
model = YOLO('best_incomplete_training.pt')

# Load video 
cap = cv2.VideoCapture('moving_truck.mp4')

# Get width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define a codec and create a video writer object to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Define line position
line_y = 130

# Font settings for status text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 2 
font_color = (0, 0, 255) 

# Loop until video is fully processed
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        # Exit loop if no frame is read (end of video)
        break

    # Draw the line on the frame
    cv2.line(frame, (250, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)

    # Detect objects in the frame
    results = model(frame)

    # Parse detection results
    for result in results:
        # Get bounding boxes from the result
        boxes = result.boxes
        classes = result.names

        # Debug information for classes and boxes
        print(f"Classes: {classes}")
        print(f"Boxes: {boxes}")

        for box in boxes:
            # Extract bounding box coordinates and other information
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            # Get the class label for the detected object
            class_label = classes[cls]

            # Calculate the center coordinates of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Check if the bounding box intersects with the line
            if class_label == "truck":
                if y1 <= line_y <= y2:
                    status = "in"  # Truck is crossing the line
                else:
                    status = "out"  # Truck is not crossing the line
            else:
                status = "out2"  # Non-truck is considered "out"

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate text size to center it properly
            text_size, _ = cv2.getTextSize(status, font, font_scale, font_thickness)
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2

            # Put the status text at the center of the bounding box
            cv2.putText(frame, status, (text_x, text_y), font, font_scale, font_color, font_thickness)

            # Debug information
            print(f"Box: [{x1}, {y1}, {x2}, {y2}], Center: ({center_x}, {center_y}), Status: {status}, Class: {class_label}")

    # Write the processed frame to the output video
    out.write(frame)

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
