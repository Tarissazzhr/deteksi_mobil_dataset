from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov10n.pt')  

# Train the model on your dataset
model.train(data='C:/Users/Tarisa/Visual Studio Code/projekk2/dataset/data.yaml', epochs=10, imgsz=640, workers=1, verbose=True)
