import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('C:/Users/Tarisa/Visual Studio Code/projekk2/yolov8s.pt')

# Load an image or video frame
image = cv2.imread('C:/Users/Tarisa/Visual Studio Code/projekk2/dataset/test/images/DOH_mp4-14_jpg.rf.7eda23f94fab050a34c1a74bc7760f95.jpg')

# Perform inference
results = model(image)

# Extract detection results
detections = results[0].boxes

# Draw bounding boxes and count cars
car_count = 0
for box in detections:
    # Check if 'car' class is labeled as 0
    if int(box.cls) == 0:  # Ensure this is the correct attribute
        car_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy)  # Convert to integers
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Display the image with detections
cv2.imshow('Car Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Number of cars detected: {car_count}")
