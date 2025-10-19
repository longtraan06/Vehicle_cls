from ultralytics import YOLO

# Load mô hình
model = YOLO("/home/rmie/Desktop/Workspaces/Vehicle_Cls/best.pt")

# In danh sách class
print(model.names)
