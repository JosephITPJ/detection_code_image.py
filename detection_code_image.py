import cv2
import json
import traceback
import tkinter as tk
from tkinter import file dialog, messagebox
from PIL import Image, ImageTk

def load_model():

        model = cv2.dnn_DetectionModel("files/yolov4.weights", "files/yolov4.cfg")
        model.setInputSize(320, 320)
        model.setInputScale(1.0 / 127.5)
        return model

def load_class_labels(filename='files/thingnames.txt'):
    with open(filename, 'rt') as spt:
        return spt.read().rstrip('\n').split('\n')

def detect_objects(model, class_labels, image_path):
    img = cv2.imread(image_path)

    classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 2
    fontThickness = 2
    image_result = {"image_path": image_path, "detections": []}
    for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
        label = class_labels[classInd]
        detection = {"class": label, "confidence": float(conf), "bbox": boxes.tolist()}
        image_result["detections"].append(detection)
        x, y, w, h = map(int, boxes)

        img_copy = img.copy()

        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
        cv2.putText(img_copy, f'{label}: {conf:.2f}', (x, y - 10), font, fontScale, (0, 255, 0), fontThickness,
                    cv2.LINE_AA)

    image_results_list = [image_result]

    results_json = {"images": image_results_list}
    with open("detection_results.json", "w") as json_file:
        json.dump(results_json, json_file)

    # Visualize the result
    cv2.imshow('result', img_copy)
    cv2.waitKey(0)
    cv2.imwrite('result.png', img_copy)

def upload_image(model, class_labels):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        detect_objects(model, class_labels, file_path)

def main():
    # Create the main application window
    root = tk.Tk()
    root.title("Object Detection GUI")

    # Load the YOLO model and class labels
    yolo_model = load_model()
    if yolo_model is None:
        return

    class_labels = load_class_labels()

    # Create a welcome label
    welcome_label = tk.Label(root, text="Welcome to Object Detection system!", font=("Helvetica", 16))
    welcome_label.pack(pady=20)

    # Create and set up GUI components
    upload_button = tk.Button(root, text="Upload Image", command=lambda: upload_image(yolo_model, class_labels))
    upload_button.pack(pady=20)

    # Display a messagebox with instructions
    messagebox.showinfo("Instructions", "Welcome! Click 'Upload Image' to start detecting objects.")

    # Start the GUI main loop
    root.mainloop()

if __name__ == "__main__":
    main()
