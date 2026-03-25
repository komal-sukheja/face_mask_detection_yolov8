import argparse
import cv2
import time
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
CONF = 0.4
IMG_SIZE = 320


def draw_boxes(frame, results):
    boxes = results[0].boxes
    if boxes is None:
        return frame

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])

        if cls == 0:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

    return frame


def main(source, save_output=False):
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0 if source == "webcam" else source)

    if not cap.isOpened():
        print("Error: Could not open source")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        start_time = time.time()

        frame = cv2.resize(frame, (480, 360))

        results = model.predict(
            frame,
            conf=CONF,
            imgsz=IMG_SIZE,
            verbose=False
        )

        frame = draw_boxes(frame, results)

        fps = 1 / (time.time() - start_time + 1e-6)

        cv2.rectangle(frame, (10, 10), (150, 55), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        if save_output:
            if out is None:
                h, w, _ = frame.shape
                out = cv2.VideoWriter("output_test.mp4", fourcc, 20, (w, h))
            out.write(frame)

        cv2.imshow("Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="webcam")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    main(args.source, args.save)