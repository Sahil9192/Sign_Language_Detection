import os
from dataclasses import dataclass,field
from datetime import datetime
from Sign_Language_Recognition.constant.training_pipeline import *


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = os.path.join(ARTIFACTS_DIR,TIMESTAMP)
    def __post_init__(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)

training_pipeline_config:TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME
    )  

    feature_store_file_path: str = os.path.join(
        data_ingestion_dir , DATA_INGESTION_FEATURE_STORE_DIR 
    )

    data_download_url: str = DATA_DOWNLOAD_URL


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_VALIDATION_DIR_NAME
    )

    valid_status_file_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_STATUS_FILE)

    required_file_list: list = field(default_factory=lambda: DATA_VALIDATION_ALL_REQUIRED_FILES)


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, MODEL_TRAINER_DIR_NAME
    )

    weight_name: str = MODEL_TRAINER_PRETRAINED_WEIGHT_NAME
    no_epochs: str = MODEL_TRAINER_NO_EPOCHS
    batch_size: str = MODEL_TRAINER_BATCH_SIZE

    model_save_path: str = os.path.join("yolov5", "my_model.pt")  



@dataclass
class ModelPusherConfig:
    github_model_url: str = MODEL_PUSHER_GITHUB_MODEL_URL
    local_model_save_path: str = MODEL_LOCAL_SAVE_PATH

