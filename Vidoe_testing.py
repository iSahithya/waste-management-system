from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("Models/best.pt")

# Load the video file or use 0 for webcam
video_source = "path/to/video.mp4"  # Replace with your video file path or use 0 for webcam
cap = cv2.VideoCapture(video_source)

# Check if the video source is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video processing complete.")
        break

    # Perform predictions on the current frame
    results = model.predict(source=frame, conf=0.1, verbose=False)

    # Visualize the predictions on the frame
    for result in results:
        # Plot the results on the frame
        frame = result.plot()

    # Resize the frame for display (optional)
    scale_percent = 100  # Adjust as needed
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # Show the frame with predictions
    cv2.imshow("YOLO Video Predictions", resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resources and close display windows
cap.release()
cv2.destroyAllWindows()
