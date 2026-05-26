
from ultralytics import YOLO #object detection
import os
from datetime import datetime
import cv2

script_dir = os.path.dirname(os.path.realpath(__file__))
print(script_dir)

# Load a pretrained YOLO model
model = YOLO ('yolo26n.pt')

# Run inference on a specific image
#results1 = model(f"{script_dir}\\data\\line_empty.png", show=True, conf=0.01, iou=0.15,imgsz=2400)
#print (results1)

# Run inference on a specific image
#results2 = model(f"{script_dir}\\data\\line_full.jpg", show=True)
results2 = model(f"{script_dir}\\data\\line_full.jpg", show=True, conf=0.01, iou=0.15, imgsz=2400)
#Force the model to process images at a higher resolution
print (results2)
person_count = len(results2[0].boxes)
print(f"Total people detected: {person_count}")

annotated_filename = f"annotated_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
output_img_path = f"{script_dir}\\out\\{annotated_filename}"
annotated_frame = results2[0].plot()
cv2.imwrite(output_img_path, annotated_frame)
cv2.destroyAllWindows()