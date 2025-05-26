import os
import json
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
from ultralytics import YOLO
from flask import send_file
import cv2

app = Flask(__name__)

binary_model = load_model("models/binary_classifier_2.h5")
type_model_path = "models/type_classifier.h5"
type_model = load_model(type_model_path) if os.path.exists(type_model_path) else None

yolo_model_path = "models/detector.pt"
yolo_model = YOLO(yolo_model_path) if os.path.exists(yolo_model_path) else None

class_mapping_path = "utils/class_mapping.json"
class_mapping = {}
if os.path.exists(class_mapping_path):
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)

IMG_SIZE = 224

def prepare_image(img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file)
    input_tensor = prepare_image(img)

    binary_pred = binary_model.predict(input_tensor)[0][0]
    is_trash = binary_pred > 0.5
    result = "trash" if is_trash else "clean"

    class_name = None
    confidence = None
    yolo_detections = []

    if is_trash and type_model:
        type_preds = type_model.predict(input_tensor)[0]
        best_idx = np.argmax(type_preds)
        class_name = class_mapping.get(str(best_idx), f"class_{best_idx}")
        confidence = float(type_preds[best_idx])

    if is_trash and yolo_model:
        temp_path = "temp_input.jpg"
        img.save(temp_path)

        results = yolo_model.predict(source=temp_path, save=False, conf=0.1)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = class_mapping.get(str(cls_id), f"class_{cls_id}")
                conf = float(box.conf[0])
                xywh = box.xywh[0].tolist()
                yolo_detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x_center": round(xywh[0], 2),
                        "y_center": round(xywh[1], 2),
                        "width": round(xywh[2], 2),
                        "height": round(xywh[3], 2)
                    }
                })

        os.remove(temp_path)

    return jsonify({
        "result": result,
        "class": class_name,
        "confidence": round(confidence, 4) if confidence else None,
        "detections": yolo_detections
    })

def draw_detections(image_path, detections, output_path="output.jpg"):
    image = cv2.imread(image_path)
    for det in detections:
        box = det["bbox"]
        class_name = det["class"]
        conf = det["confidence"]

        x = int(box["x_center"] - box["width"] / 2)
        y = int(box["y_center"] - box["height"] / 2)
        w = int(box["width"])
        h = int(box["height"])

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{class_name} ({conf:.2f})"
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
    return output_path

@app.route("/visualize", methods=["POST"])
def visualize():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file)
    input_tensor = prepare_image(img)

    binary_pred = binary_model.predict(input_tensor)[0][0]
    is_trash = binary_pred > 0.5

    if not is_trash or not yolo_model:
        return jsonify({"error": "Image is clean or model not available"}), 400

    temp_path = "temp_input.jpg"
    output_path = "output.jpg"
    img.save(temp_path)

    results = yolo_model.predict(source=temp_path, save=False, conf=0.3)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = class_mapping.get(str(cls_id), f"class_{cls_id}")
            conf = float(box.conf[0])
            xywh = box.xywh[0].tolist()
            detections.append({
                "class": cls_name,
                "confidence": round(conf, 4),
                "bbox": {
                    "x_center": round(xywh[0], 2),
                    "y_center": round(xywh[1], 2),
                    "width": round(xywh[2], 2),
                    "height": round(xywh[3], 2)
                }
            })

    draw_detections(temp_path, detections, output_path=output_path)
    return send_file(output_path, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
