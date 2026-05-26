

#v2 - count objects
#v1 - load image - object detection

from ultralytics import YOLO #object detection
from ultralytics import solutions # object count
import os
import cv2

# Get the current working directory
script_dir = os.path.dirname(os.path.realpath(__file__))
print(script_dir)

# Load a pretrained YOLO model (e.g., yolo26n)
model = YOLO ('yolo26n.pt')

# Evaluate the model's performance on the validation set
#results = model.val()

# Perform object detection on an image using the model
# Run inference on a specific image
#results1 = model(f"{script_dir}\\line_empty.png", show=True)
#print (results1)

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
    # region=region_points,  # pass region points
    model="yolo26n.pt",  # model="yolo26n-obb.pt" for object counting with OBB model.
    #classes=[0, 2],  # count specific classes, e.g., person and car with the COCO pretrained model.
    classes=[0],  # count specific classes, e.g., person with the COCO pretrained model.
    # tracker="botsort.yaml",  # choose trackers, e.g., "bytetrack.yaml"
)

# Run inference on a specific image
#results2 = model(f"{script_dir}\\line_full.jpg", show=True)

# Run inference on an image, filtering for class 0 (person)
results2 = model(f"{script_dir}\\line_full.jpg", show=True, classes=[0])

#results2 = counter(f"{script_dir}\\line_full.jpg")
print (results2)

# Get the count of detected persons
person_count = len(results2[0].boxes)
print(f"Total people detected: {person_count}")

# 5. Save the annotated output image
output_img_path = f"{script_dir}\\output_annotated.jpg"
annotated_frame = results2[0].plot()
cv2.imwrite(output_img_path, annotated_frame)
print(f"Annotated image saved to: {output_img_path}")
