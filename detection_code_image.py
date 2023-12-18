import cv2
import json

config_file = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'files/frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320) #greater this value better the reults tune it for best output
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

classLabels = []
filename = 'files/thingnames.txt'
with open(filename, 'rt') as spt:
    classLabels = spt.read().rstrip('\n').split('\n')



# Placeholder for the list to store results
image_results_list = []


img = cv2.imread('images/test_image.png')

classIndex, confidence, bbox = model.detect(img, confThreshold=0.6) #tune confThreshold for best results

font = cv2.FONT_HERSHEY_PLAIN

image_result = {"image_path": 'images/test_image.png', "detections": []}
for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    label = classLabels[classInd - 1]  # Adjust for 0-based indexing
    detection = {"class": label, "confidence": float(conf), "bbox": boxes.tolist()}
    image_result["detections"].append(detection)

# Append results for the current image to the list
image_results_list.append(image_result)

# Save the results in a JSON file
results_json = {"images": image_results_list}
with open("detection_results.json", "w") as json_file:
    json.dump(results_json, json_file)

# Visualize the result
for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=3, color=(0, 255, 0), thickness=3)

cv2.imshow('result', img)
cv2.waitKey(0)
cv2.imwrite('result.png', img)