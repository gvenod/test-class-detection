

#v3 - distance calculator
#v2 - count objects
#v1 - load image - object detection

from ultralytics import YOLO #object detection
from ultralytics import solutions # object count
import os
import cv2
import numpy as np

from datetime import datetime

import ultralytics
ultralytics.checks()


# Get the current working directory
script_dir = os.path.dirname(os.path.realpath(__file__))
print(script_dir)

# Load a pretrained YOLO model (e.g., yolo26n)
model = YOLO ('yolo26n.pt')

# Evaluate the model's performance on the validation set
#results = model.val()

# Perform object detection on an image using the model
# Run inference on a specific image
#results1 = model(f"{script_dir}\\out\\line_empty.png", show=True)
#print (results1)



import random

def generate_yolo_polygon_points(num_points=4, img_width=1280, img_height=720):
    """
    Generates random region points for YOLO polygon zones.
    Ensures points are in a logical sequence to form a valid polygon.
    """
    # Generate random points
    points = [(random.randint(0, img_width), random.randint(0, img_height)) for _ in range(num_points)]
    
    # Sort points by angle around their centroid to prevent overlapping lines
    cx = sum(p[0] for p in points) / num_points
    cy = sum(p[1] for p in points) / num_points
    points.sort(key=lambda p: (p[0] - cx, p[1] - cy))
    
    return points



frame = cv2.imread(f"{script_dir}\\data\\line_full.jpg")
# Get dimensions
dimensions = frame.shape

# Unpack for specific values
height = frame.shape[0]
width = frame.shape[1]

# Generate a 4-point bounding region for a 1920x1080 frame
region_points = generate_yolo_polygon_points(num_points=2, img_width=width, img_height=height)

# Format for YOLO (e.g., for Ultralytics solutions)
formatted_points = [list(pt) for pt in region_points]

print("Generated YOLO Region Points:")
print(formatted_points)
print(f"Generated {len(formatted_points)} regions.")


#class names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
# 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
#20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
#32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 
#43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 
#56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 
#68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # display the output
    #region=formatted_points,  # pass region points
    model="yolo26n.pt",  # model="yolo26n-obb.pt" for object counting with OBB model.
    #classes=[0, 2],  # count specific classes, e.g., person and car with the COCO pretrained model.
    classes=[0],  # count specific classes, e.g., person with the COCO pretrained model.
    # tracker="botsort.yaml",  # choose trackers, e.g., "bytetrack.yaml"
)

# Initialize distance calculation object
#distancecalculator = solutions.DistanceCalculation(
#    model="yolo26n.pt",  # path to the YOLO26 model file.
#    show=True,  # display the output
#)

#results_dist = distancecalculator(frame)
#print (results_dist)
# Get current time and format it
#dist_filename = f"dist_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
#cv2.imwrite(f"{script_dir}\\out\\{dist_filename}", results_dist.plot_im)


results_count = counter(frame)
print (results_count)
# Get current time and format it
count_filename = f"count_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
cv2.imwrite(f"{script_dir}\\out\\{count_filename}", results_count.plot_im)


# Run inference on a specific image
#results2 = model(f"{script_dir}\\data\\line_full.jpg", show=True)

# Run inference on an image, filtering for class 0 (person)
#results2 = model(f"{script_dir}\\data\\line_full.jpg", show=True, classes=[0])
# Get the count of detected persons
#person_count = len(results2[0].boxes)
#print(f"Total people detected: {person_count}")

# 5. Save the annotated output image
#annotated_filename = f"annotated_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
#output_img_path = f"{script_dir}\\out\\{annotated_filename}"
#annotated_frame = results2[0].plot()
#cv2.imwrite(output_img_path, annotated_frame)

cv2.destroyAllWindows()
