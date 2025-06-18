import os,sys
import yaml

from Sign_Language_Recognition.utils.main_utils import read_yaml_file
from Sign_Language_Recognition.logger import logging
from Sign_Language_Recognition.exception import SignException
from Sign_Language_Recognition.entity.config_entity import ModelTrainerConfig
from Sign_Language_Recognition.entity.artifacts_entity import ModelTrainerArtifact
from Sign_Language_Recognition.utils.main_utils import get_latest_yolov5_best_model_path

class ModelTrainer:
    def __init__(
            self,
            model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config

    
    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            os.system("unzip Sign_language_data.zip")
            os.system("rm Sign_language_data.zip")

            with open("data.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")

            config['nc'] = int(num_classes)


            with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)

            os.system(f"cd yolov5/ && python train.py --img 416 --batch {self.model_trainer_config.batch_size} --epochs {self.model_trainer_config.no_epochs} --data ../data.yaml --cfg ./models/custom_yolov5s.yaml --weights {self.model_trainer_config.weight_name} --name yolov5s_results  --cache")
            best_model_path = get_latest_yolov5_best_model_path()
            final_model_path = os.path.join("yolov5", "my_model.pt")
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            os.system(f"cp {best_model_path} {final_model_path}")

            best_model_path = get_latest_yolov5_best_model_path()  # This dynamically finds latest best.pt
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            os.system(f"cp {best_model_path} {final_model_path}")


            # Optional: also copy to artifact dir
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            artifact_model_path = os.path.join(self.model_trainer_config.model_trainer_dir, "my_model.pt")
            os.system(f"cp {final_model_path} {artifact_model_path}")

           
            os.system("rm -rf yolov5/runs")
            os.system("rm -rf train")
            os.system("rm -rf test")
            os.system("rm -rf data.yaml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=final_model_path
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise SignException(e, sys)