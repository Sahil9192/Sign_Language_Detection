import sys, os
from Sign_Language_Recognition.logger import logging
from Sign_Language_Recognition.exception import SignException
from Sign_Language_Recognition.components.data_ingestion import DataIngestion
from Sign_Language_Recognition.components.data_validation import DataValidation
from Sign_Language_Recognition.components.model_trainer import ModelTrainer
from Sign_Language_Recognition.components.model_pusher import ModelPusher
from Sign_Language_Recognition.entity.config_entity import (DataIngestionConfig, DataValidationConfig, ModelTrainerConfig, ModelPusherConfig)
from Sign_Language_Recognition.entity.artifacts_entity import (DataIngestionArtifact,DataValidationArtifact, ModelTrainerArtifact,ModelPusherArtifact)
from Sign_Language_Recognition.utils.main_utils import get_latest_yolov5_best_model_path
import shutil
import glob

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self)-> DataIngestionArtifact:
        try:
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )
            logging.info("Getting the data from URL")

            data_ingestion = DataIngestion(
                data_ingestion_config =  self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from URL")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifact
        
        except Exception as e:
            raise SignException(e,sys)
        
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise SignException(e, sys) from e
        

    def start_model_trainer(self) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise SignException(e, sys) from e
            
    def start_model_pusher(self, model_trainer_artifact: ModelTrainerArtifact):
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_trainer_artifact=model_trainer_artifact
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact

        except Exception as e:
            raise SignException(e, sys)
        
    

    def get_latest_yolov5_best_model_path():
        weight_paths = glob.glob("yolov5/runs/train/yolov5s_results*/weights/best.pt")
        if not weight_paths:
            raise FileNotFoundError("No YOLOv5 trained model found in yolov5/runs/train/")
        return max(weight_paths, key=os.path.getctime)

        
    
    
        
    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
        )

            if data_validation_artifact.validation_status == True:
                model_trainer_artifact = self.start_model_trainer()
                model_pusher_artifact = self.start_model_pusher(
                    model_trainer_artifact=model_trainer_artifact
            )

            # 🔍 Get latest YOLOv5 trained model path dynamically
                source_model_path = get_latest_yolov5_best_model_path()

                destination_model_path = os.path.join("yolov5", "my_model.pt")
                os.makedirs(os.path.dirname(destination_model_path), exist_ok=True)
                shutil.copy(source_model_path, destination_model_path)

                logging.info(f"Copied trained model from {source_model_path} to {destination_model_path}")

            else:
                raise Exception("Data validation failed. Model training cannot be performed.")

        except Exception as e:
            raise SignException(e, sys)


