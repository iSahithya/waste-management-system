from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("Models/best.pt")

# Perform predictions on a test image
results = model.predict(source="Plastic Bags.v1i.yolov8/test/images/00000007_jpg.rf.c5e37a215787e78b88a1bae1208d78c0.jpg", conf=0.1)

# Prepare a dictionary to store detected categories
detected_categories = {}

# Process the results
for result in results:
    # Extract predictions from the result
    boxes = result.boxes  # Bounding boxes and associated information

    # Iterate through each detected object
    for box in boxes:
        # Extract class index and confidence score
        class_id = int(box.cls[0])
        confidence = box.conf[0]

        # Get class name using the model's class names
        class_name = model.names[class_id]

        # Categorize and add detected class names based on conditions
        if class_name == "Battery" or class_name == "Medicine":
            category = "Hazardous Waste"
        elif class_name == "cardboard" or class_name == "Metal" or class_name == "plastic_bottle":
            category = "Recyclable Waste"
        elif class_name == "food_waste":
            category = "Organic Waste"
        elif class_name == "plastic_bags":
            category = "Non-Recyclable Waste"
        else:
            continue  # Skip if it doesn't match any category

        # Add to detected categories
        if category not in detected_categories:
            detected_categories[category] = []
        detected_categories[category].append(f"{class_name} ({confidence:.2f})")

    # Visualize the predictions on the image
    img = result.plot()

    # Add detected categories to the image
    y_offset = 20  # Initial offset for the text
    x_offset = 10  # Fixed left margin

    for category, items in detected_categories.items():
        # Write category name
        cv2.putText(img, category, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 20

        # Write detected items under the category
        for item in items:
            cv2.putText(img, f"- {item}", (x_offset + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 15

    # Resize the output image for display
    scale_percent = 100  # Adjust this percentage to scale the image
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # Show the resized image
    cv2.imshow("YOLO Predictions", resized_img)

    # Wait for the 'q' key to be pressed
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
