import cv2
config_file = 'Assets/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
model = 'Assets/coco-detection-main/frozen_inference_graph.pb'
final = cv2.dnn_DetectionModel(model,config_file)
classes = []
filename = 'Assets/yolo3.txt'

with open(filename,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
print(classLabels)