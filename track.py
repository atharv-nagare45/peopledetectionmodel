from ultralytics import YOLO
import cv2
import os

MODEL_PATH = r"runs\detect\water_rescue_person\person_model_gpu-2\weights\best.pt"
TRACKER_PATH = r"trackers\custom_bytetrack.yaml"

if not os.path.exists(MODEL_PATH):
    print("ERROR: best.pt not found!")
    print("Expected path:")
    print(os.path.abspath(MODEL_PATH))
    exit()

if not os.path.exists(TRACKER_PATH):
    print("ERROR: custom_bytetrack.yaml not found!")
    print("Expected path:")
    print(os.path.abspath(TRACKER_PATH))
    exit()

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully!")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened!")
    exit()

print("Camera started successfully!")
print("Press Q to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER_PATH,
        conf=0.30,
        verbose=False
    )

    result = results[0]
    annotated_frame = result.plot()

    if result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()

        for track_id in track_ids:
            print(f"Person Track ID: {track_id}")

    cv2.imshow(
        "Smart Rescue - YOLO + ByteTrack",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Tracking stopped.")
