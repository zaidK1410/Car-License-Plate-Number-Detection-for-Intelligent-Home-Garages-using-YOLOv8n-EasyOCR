import os
import cv2
import easyocr
from ultralytics import YOLO

# -----------------------------
# Paths (relative for GitHub)
# -----------------------------
MODEL_PATH = "models/best.pt"
INPUT_FOLDER = "data/images"
OUTPUT_FOLDER = "data/predictions"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Load model + OCR
# -----------------------------
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'])

# -----------------------------
# Inference
# -----------------------------
for img_name in os.listdir(INPUT_FOLDER):

    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue


    img_path = os.path.join(INPUT_FOLDER, img_name)
    img = cv2.imread(img_path)

    results = model(img, verbose=False)

    boxes = results[0].boxes.xyxy.cpu().numpy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        crop = img[y1:y2, x1:x2]

        ocr_result = reader.readtext(crop, allowlist='0123456789')

        text = "".join([t for (_, t, conf) in ocr_result])

        print(f"{img_name} → Plate {i}: {text}")

        cv2.imwrite(
            os.path.join(OUTPUT_FOLDER, f"{img_name}_crop_{i}.jpg"),
            crop
        )

    results[0].save(os.path.join(OUTPUT_FOLDER, img_name))

