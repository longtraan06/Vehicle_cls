from ultralytics import YOLO

# Load model PyTorch
model = YOLO("/home/rmie/Desktop/Workspaces/Vehicle_Cls/newdata.pt")

model.export(format="ncnn", imgsz=640)  
# ncnn_model = YOLO("yolo11n_ncnn_model")

# results = ncnn_model("image.jpg")
