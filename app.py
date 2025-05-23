import sys
import os
import uuid
from Sign_Language_Recognition.pipeline.training_pipeline import TrainPipeline
from Sign_Language_Recognition.exception import SignException
from Sign_Language_Recognition.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
import subprocess

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
        return "Training Successful!!"
    except Exception as e:
        return f"Training failed: {str(e)}", 500

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']

        # Create a unique filename
        unique_filename = f"inputImage_{uuid.uuid4().hex}.jpg"

        # Save inside yolov5 folder so detect.py can find it easily
        yolov5_folder = os.path.join(os.getcwd(), "yolov5")
        input_path = os.path.join(yolov5_folder, unique_filename)

        decodeImage(image, input_path)

        # Run detection using subprocess
        detect_cmd = [
            "python", os.path.join(yolov5_folder, "detect.py"),
            "--weights", os.path.join(yolov5_folder, "yolov5s.pt"),
            "--img", "416",
            "--conf", "0.5",
            "--source", input_path
        ]
        subprocess.run(detect_cmd, check=True)

        # Output image path - yolov5 saves outputs here with same filename
        output_path = os.path.join(yolov5_folder, "runs", "detect", "exp", unique_filename)

        # Encode output image to base64
        opencodedbase64 = encodeImageIntoBase64(output_path)

        # Clean up input and output files
        if os.path.exists(input_path):
            os.remove(input_path)
        # Remove the whole 'exp' folder after processing to keep things clean
        exp_folder = os.path.join(yolov5_folder, "runs", "detect", "exp")
        if os.path.exists(exp_folder):
            subprocess.run(["rm", "-rf", exp_folder], check=True)

        result = {"image": opencodedbase64.decode('utf-8')}
        return jsonify(result)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return Response(f"Prediction failed: {str(e)}", status=500)

@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        yolov5_folder = os.path.join(os.getcwd(), "yolov5")
        subprocess.run([
            "python", os.path.join(yolov5_folder, "detect.py"),
            "--weights", os.path.join(yolov5_folder, "yolov5s.pt"),
            "--img", "416",
            "--conf", "0.5",
            "--source", "0"
        ], check=True)

        # Clean up runs folder after live detection stops
        runs_folder = os.path.join(yolov5_folder, "runs")
        if os.path.exists(runs_folder):
            subprocess.run(["rm", "-rf", runs_folder], check=True)
        return "Camera starting!!"

    except Exception as e:
        print(f"Error starting live detection: {e}")
        return Response(f"Failed to start live detection: {str(e)}", status=500)

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080, debug=True)
