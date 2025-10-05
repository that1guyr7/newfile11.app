import cv2
config_file = 'Assets/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
model = 'Assets/coco-detection-main/frozen_inference_graph.pb'
final = cv2.dnn_DetectionModel(model,config_file)
classes = []
filename = 'Assets/yolo3.txt'

with open(filename,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
print(classLabels)

final.setInputSize(320,320)
final.setInputScale(1.0/127.5)
final.setInputMean((127.5,127.5,127.5))
final.setInputSwapRB(True)
sample = cv2.imread("Assets/download.jpeg")
index,confidence,bbox = final.detect(sample,confThreshold=0.5)
print(confidence,index)

for i,c,b in zip(index.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(sample,b,(255,0,0),2)
    cv2.putText(sample,classLabels[i-1],(b[0]+40,b[1]+40),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),1)

cv2.imshow('final',sample)
cv2.waitKey(0)
