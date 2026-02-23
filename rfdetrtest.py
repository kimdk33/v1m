import time
import cv2
import supervision as sv
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

model=RFDETRNano()
model.optimize_for_inference()

tracker= sv.ByteTrack()

file_path = "./videos/test2.mp4"
cctv_id = "rfdetr_test2"

cap = cv2.VideoCapture(file_path)

fps = cap.get(cv2.CAP_PROP_FPS)
one_second = 1*16/fps

while cap.isOpened():
    ret, frame = cap.read()

    if not ret: break

    frame[:200, :] = 0
    
    start_time = time.perf_counter()
    results = model.predict(frame, 0.5) 
    print(f"detection 끝: {time.perf_counter() - start_time}")
    results = tracker.update_with_detections(results)
    print(f"tracker 끝: {time.perf_counter() - start_time}")

    # labels = [f"{COCO_CLASSES[class_id]}" for class_id in results.class_id]