# 🚗 License Plate Detection for Smart Home Garage

Automatic License Plate Recognition (ALPR) system built using YOLOv8n and EasyOCR.

This project detects car license plates in real time and extracts the plate number to allow or deny garage access. Designed for low-cost edge devices like Raspberry Pi.

---

## ✨ Features
- Real-time license plate detection
- OCR number recognition
- Webcam/IP camera support
- Access control (allowed / denied)
- Lightweight for edge devices

---

## 🛠 Tech Stack
- Python
- YOLOv8 (Ultralytics)
- EasyOCR
- OpenCV
- NumPy

---

## 📂 Project Structure
license-plate-detection/

│
├── models/
│   └── best.pt
│
├── data/
│   ├── images/
│   └── predictions/
│
├── main.py
├── requirements.txt
