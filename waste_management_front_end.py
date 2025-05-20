from flask import Flask, request, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load the trained model
model = YOLO("Models/best.pt")

UPLOAD_FOLDER = "static/uploads/"
RESULT_FOLDER = "static/results/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No file selected")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Perform object detection
        results = model.predict(source=file_path, conf=0.1)

        # Prepare dictionary for detected categories
        detected_categories = {}

        for result in results:
            boxes = result.boxes  # Extract detected objects

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                class_name = model.names[class_id]

                # Categorize detected objects
                if class_name in ["Battery", "Medicine"]:
                    category = "Hazardous Waste"
                elif class_name in ["cardboard", "Metal", "plastic_bottle"]:
                    category = "Recyclable Waste"
                elif class_name == "food_waste":
                    category = "Organic Waste"
                elif class_name == "plastic_bags":
                    category = "Non-Recyclable Waste"
                else:
                    continue

                if category not in detected_categories:
                    detected_categories[category] = []
                detected_categories[category].append(f"{class_name} ({confidence:.2f})")

            # Visualize detections on the image
            img = result.plot()
            result_path = os.path.join(RESULT_FOLDER, file.filename)
            cv2.imwrite(result_path, img)

        return render_template("index.html", image=result_path, categories=detected_categories)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
