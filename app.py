from Sign_Language_Recognition.logger import logging
from Sign_Language_Recognition.exception import SignException
import sys
from Sign_Language_Recognition.pipeline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()