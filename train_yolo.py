from datetime import datetime

from ultralytics import YOLO

ddhh = datetime.now().strftime("%y%m%d-%H%M%S")
MODEL_NAME = 'yolov8m-seg.pt'   # .pt loads a pretrained model
DATA_PATH = 'yolo_config.yaml'

model = YOLO(MODEL_NAME)
results = model.train(data=DATA_PATH, epochs=100, batch=4, imgsz=640,
                        save=True, cache=True, lr0=1e-3, lrf=1e-4,
                        name=f'{ddhh}-glaucoma-{MODEL_NAME.split(".")[0]}')
results = model.val()