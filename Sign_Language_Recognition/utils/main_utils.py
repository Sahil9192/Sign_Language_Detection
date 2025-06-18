import os.path
import sys
import yaml
import base64
import re
from Sign_Language_Recognition.exception import SignException
from Sign_Language_Recognition.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            logging.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise SignException(e, sys) from e
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)
            logging.info("Successfully write_yaml_file")

    except Exception as e:
        raise SignException(e, sys)
    



def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./data/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    


def get_latest_yolov5_best_model_path(yolov5_runs_dir="yolov5/runs/train"):
    """
    Finds the most recent 'best.pt' model path from the YOLOv5 runs/train/ directory.
    Returns the full path to best.pt or raises an exception if not found.
    """
    try:
        if not os.path.exists(yolov5_runs_dir):
            raise FileNotFoundError(f"{yolov5_runs_dir} does not exist")

        # Get list of exp directories (e.g., exp, exp1, exp2)
        exp_dirs = [os.path.join(yolov5_runs_dir, d) for d in os.listdir(yolov5_runs_dir)
                    if os.path.isdir(os.path.join(yolov5_runs_dir, d)) and re.match(r"exp\d*$", d)]

        if not exp_dirs:
            raise FileNotFoundError("No experiment folders found in yolov5/runs/train/")

        # Find the latest exp*/ folder by creation time
        latest_exp = max(exp_dirs, key=os.path.getctime)
        best_model_path = os.path.join(latest_exp, "weights", "best.pt")

        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"best.pt not found at: {best_model_path}")

        return best_model_path

    except Exception as e:
        raise SignException(e, sys)
