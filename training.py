from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Train the model
train_results = model.train(
    data="./datasets/data.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    # device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    workers=4,
    batch=16,
    name="military_tanks_detection",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("./test-tank.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
