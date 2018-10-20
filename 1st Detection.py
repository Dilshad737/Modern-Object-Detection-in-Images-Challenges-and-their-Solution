import tensorflow as tf
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()

execution_path = os.path.join(execution_path, "MODELS/DETECTION/yolo.h5")
test = os.getcwd()
detector.setModelPath(execution_path)
detector.loadModel()
detections = (detector.detectObjectsFromImage(input_image=os.path.join(test , "11.jpg"),
                                             output_image_path=os.path.join(test , "imagenew_11.jpg")))


for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
