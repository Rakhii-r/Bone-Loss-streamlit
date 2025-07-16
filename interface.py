import gradio as gr
from ultralytics import YOLO

model = YOLO(r"C:\Users\saket\OneDrive\Desktop\don\best (1).pt")

def detect(image):
    results = model.predict(image, conf=0.25)
    return results[0].plot()

gr.Interface(fn=detect, inputs="image", outputs="image").launch()
