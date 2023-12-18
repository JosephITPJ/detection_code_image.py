import cv2
import json
import traceback
try:
    model = cv2.dnn_DetectionModel("files/yolov4.weights", "files/yolov4.cfg")
    model.setInputSize(320, 320)
    model.setInputScale(1.0/127.5)

except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()
    
classLabels = []
filename = 'files/thingnames.txt'
with open(filename, 'rt') as spt:
    classLabels = spt.read().rstrip('\n').split('\n')

# Placeholder for the list to store results
image_results_list = []

img = cv2.imread('images/apple.jpg')

classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)  # Tune confThreshold for best results

font = cv2.FONT_HERSHEY_PLAIN
fontScale = 2
fontThickness = 2
image_result = {"image_path": 'images/apple.jpg', "detections": []}
for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    label = classLabels[classInd]  # Corrected for 0-based indexing
    detection = {"class": label, "confidence": float(conf), "bbox": boxes.tolist()}
    image_result["detections"].append(detection)
    x, y, w, h = map(int, boxes)

    # Create a copy of the image to avoid modifying the original image
    img_copy = img.copy()

    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
    cv2.putText(img_copy, f'{label}: {conf:.2f}', (x, y - 10), font, fontScale, (0, 255, 0), fontThickness, cv2.LINE_AA)

# Append results for the current image to the list
image_results_list.append(image_result)

# Save the results in a JSON file
results_json = {"images": image_results_list}
with open("detection_results.json", "w") as json_file:
    json.dump(results_json, json_file)

# Visualize the result
cv2.imshow('result', img_copy)
cv2.waitKey(0)
cv2.imwrite('result.png', img_copy)
