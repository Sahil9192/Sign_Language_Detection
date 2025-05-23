import sys
from Sign_Language_Recognition.entity.artifacts_entity import ModelPusherArtifact, ModelTrainerArtifact
from Sign_Language_Recognition.entity.config_entity import ModelPusherConfig
from Sign_Language_Recognition.exception import SignException
from Sign_Language_Recognition.logger import logging
import shutil
import os


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_trainer_artifact: ModelTrainerArtifact):
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifact = model_trainer_artifact

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher

        Description :   This method copies the trained model to a local folder
                        from where you can manually push it to GitHub.

        Output      :   Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            
            source_path = self.model_trainer_artifact.trained_model_file_path
            destination_path = self.model_pusher_config.local_model_save_path
            
            
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)

            shutil.copy2(src=source_path, dst=destination_path)
            logging.info(f"Copied trained model from {source_path} to {destination_path}")

           

            model_pusher_artifact = ModelPusherArtifact(
                github_model_url=self.model_pusher_config.github_model_url,  
                local_model_save_path=destination_path,
            )
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise SignException(e, sys) from e
