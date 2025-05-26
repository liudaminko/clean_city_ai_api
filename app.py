import os
import json
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
from ultralytics import YOLO

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

        results = yolo_model.predict(source=temp_path, save=False, conf=0.3)

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

if __name__ == "__main__":
    app.run(debug=True)
