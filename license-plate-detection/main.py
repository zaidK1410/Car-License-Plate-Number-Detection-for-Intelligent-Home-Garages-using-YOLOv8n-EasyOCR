import os
import cv2
import time
import easyocr
from ultralytics import YOLO

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/best.pt"
INPUT_FOLDER = "data/images"
OUTPUT_FOLDER = "data/predictions"
LOG_FILE = "predictions.txt"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Load model + OCR
# -----------------------------
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'])

# open log file once
log = open(LOG_FILE, "w")

# -----------------------------
# Inference
# -----------------------------
for img_name in os.listdir(INPUT_FOLDER):

    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    start = time.time()

    img_path = os.path.join(INPUT_FOLDER, img_name)
    img = cv2.imread(img_path)

    results = model(img, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    detected_texts = []

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]

        # -------------------------
        # OCR with confidence filter
        # -------------------------
        ocr_result = reader.readtext(crop, allowlist='0123456789')

        text = "".join(
            t for (_, t, conf) in ocr_result if conf >= 0.4
        )

        if text:
            detected_texts.append(text)

        # save crop
        cv2.imwrite(
            os.path.join(OUTPUT_FOLDER, f"{img_name}_crop_{i}.jpg"),
            crop
        )

    # Save full image with boxes
    results[0].save(os.path.join(OUTPUT_FOLDER, img_name))

    # -------------------------
    # Write to txt file
    # -------------------------
    if detected_texts:
        final_text = ",".join(detected_texts)
    else:
        final_text = "NOT_FOUND"

    log.write(f"{img_name} -> {final_text}\n")

    print(f"{img_name} -> {final_text} | {time.time()-start:.2f}s")

log.close()
print("All predictions saved to predictions.txt")
