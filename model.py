from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    model=YOLO("yolov8m.pt")
    
    results = model.train(data='data.yaml', epochs=100, imgsz=640, batch=8)