# 🚀Sign_Language_Detection
T- his project aims to bridge communication barriers by detecting and recognizing hand gestures from sign language using computer vision and deep learning. Leveraging a camera and the YOLOv5 object detection model, it enables real-time gesture recognition for more inclusive interaction.

---

## 🚀 Features

- 🔍 Real-time sign language detection using webcam
- 🧠 Trained with YOLOv5 for fast and accurate gesture recognition
- 📸 Support for both image-based and live video input
- 🖥️ Flask-based web interface for interaction
- 🔧 Modular code structure with reusable components

---

## Workflows

- constants
- config_entity
- artifact_entity
- components
- pipeline
- app.py

---

## 📁 Project Structure
```bash
- Sign_Language_Detection/
│
├── app.py # Flask app entry point
├── requirements.txt # Project dependencies
├── templates/
│ └── index.html # Frontend HTML
│
├── yolov5/ # YOLOv5 model & scripts
│
├── Sign_Language_Recognition/
│ ├── pipeline/ # Training pipeline
│ ├── components/ # Data/model components
│ ├── config_entity/ # Configuration classes
│ ├── artifact_entity/ # Artifact structures
│ └── utils/ # Utility functions
│
└── README.md # Project documentation

```
## 🛠️How to run:

```bash
conda create -n signlang python=3.7 -y
```


```bash
conda activate signlang
```


```bash
pip install -r requirements.txt
```

```bash
python app.py
```

```bash
Visit: http://localhost:8080
```

## 📸 Application Modes

### 📷 Predict from Uploaded Image
- Upload a base64 encoded image to /predict endpoint.
- Returns image with predicted gesture annotations.

### 🎥 Real-time Detection via Webcam
- Access the /live route to start live detection from the webcam.
- Also supports /video_feed for real-time stream via browser.

### 📦 Model Information
- Default YOLOv5 model used: yolov5s.pt
- To train a custom model, trigger training from /train route or use your own labeled dataset and modify pipeline accordingly.

### ✍️ Future Improvements
- Add support for more sign languages and gestures
- Improve detection accuracy with larger custom datasets
- Support for mobile or edge devices using TensorRT/ONNX

### 🤝 Contributions
- Pull requests are welcome. 
- For major changes, please open an issue first to discuss what you would like to change.

### 📄 License
- This project is licensed under the MIT License - see the LICENSE file for details.

```bash
Let me know if you'd like this tailored for GitHub Pages, Google Colab, or as a research repository with citation support.

```