import sys
import os
import uuid
import base64
import glob
from config import ModelConfig
import shutil
from Sign_Language_Recognition.pipeline.training_pipeline import TrainPipeline
from Sign_Language_Recognition.exception import SignException
from Sign_Language_Recognition.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
import subprocess
import traceback

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train")
def trainRoute():
    try:
        obj = TrainPipeline()
        obj.run_pipeline()

        # ✅ Copy trained model (best.pt) to yolov5/my_model.pt
        source_model_path = os.path.join("artifacts", "model_trainer", "best.pt")
        destination_model_path = os.path.join("yolov5", "my_model.pt")
        if not os.path.exists("yolov5"):
            raise FileNotFoundError("yolov5 folder not found. Please clone it before training.")

        shutil.copy(source_model_path, destination_model_path)

        return "Training Successful! Model copied to yolov5/my_model.pt"
    except Exception as e:
        return f"Training failed: {str(e)}", 500


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        data = request.get_json()
        image_data = data['image']
        image_bytes = base64.b64decode(image_data)

        os.makedirs("input_images", exist_ok=True)
        filename = f"inputImage_{uuid.uuid4().hex}.jpg"
        image_path = os.path.join("input_images", filename)
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # Run YOLOv5 detection
        relative_path = image_path.replace("\\", "/")
        command = [
            sys.executable,  # uses the active Python interpreter path
            os.path.join("yolov5", "detect.py"),
            "--weights", os.path.join("yolov5", "my_model.pt"),
            "--img", "416",
            "--conf", "0.25",
            "--source", relative_path,
            "--save-txt",
            "--save-conf",
            "--project", "runs",
            "--name", "detect",
            "--exist-ok"
]

        output_path = os.path.join("runs", "detect")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)


        subprocess.run(command, check=True)

        # Search for latest image
        output_dir = os.path.join("runs", "detect")
        image_files = glob.glob(os.path.join(output_dir, "exp*", "*.jpg"))

        # If no exp*/ image found, search in detect/ directly (fallback)
        if not image_files:
            image_files = glob.glob(os.path.join(output_dir, "*.jpg"))

        if not image_files:
            raise FileNotFoundError("No output image found in YOLO runs folder.")

        output_image_path = max(image_files, key=os.path.getctime)

        with open(output_image_path, "rb") as out_img:
            result_bytes = out_img.read()

        result_base64 = base64.b64encode(result_bytes).decode("utf-8")
        return jsonify({"image": result_base64})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500



@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        yolov5_folder = os.path.join(os.getcwd(), "yolov5")

        subprocess.Popen([
            "python", os.path.join(yolov5_folder, "detect.py"),
            "--weights", os.path.join(yolov5_folder, "yolov5s.pt"),
            "--img", "416",
            "--conf", "0.5",
            "--source", "0"
        ])

        # Optional cleanup after live detection starts (if applicable)
        runs_folder = os.path.join(yolov5_folder, "runs")
        if os.path.exists(runs_folder):
            shutil.rmtree(runs_folder)

        return "Camera started in a new window!"

    except Exception as e:
        print(f"Error starting live detection: {e}")
        return Response(f"Failed to start live detection: {str(e)}", status=500)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "API is up and running"}), 200

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8081, debug=True)
