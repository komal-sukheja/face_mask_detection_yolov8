from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict(
    source="data/face_mask/images/test",  # folder or image
    conf=0.4,
    show=True,
    save=True
)