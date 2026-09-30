from pathlib import Path
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "runs" / "detect" / "water_rescue_person" / "person_model_gpu-2" / "weights" / "best.pt"
TRACKER_PATH = BASE_DIR / "trackers" / "custom_bytetrack.yaml"

print("Loading model...")

if not MODEL_PATH.exists():
    print("ERROR: Model not found:")
    print(MODEL_PATH)
    raise SystemExit

if not TRACKER_PATH.exists():
    print("ERROR: Tracker file not found:")
    print(TRACKER_PATH)
    raise SystemExit

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("Starting camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

print("Camera started.")
print("Press Q to quit.")

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    results = model.track(
        source=frame,
        conf=0.50,
        imgsz=640,
        tracker=str(TRACKER_PATH),
        persist=True,
        verbose=False
    )

    result = results[0]

    person_count = 0
    track_ids = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id == 0:

                person_count += 1

                if box.id is not None:
                    track_id = int(box.id[0])
                    track_ids.append(track_id)

    output = result.plot()

    cv2.rectangle(
        output,
        (10, 10),
        (360, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        f"Persons: {person_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"IDs: {track_ids}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Smart Rescue - YOLO + ByteTrack",
        output
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Program closed.")
