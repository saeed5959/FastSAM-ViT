from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("yolov8x-seg.pt")

# from ndarray
im2 = cv2.imread("./dataset/000000000139.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
