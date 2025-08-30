from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience import logger
from pathlib import Path

STAGE_NAME="Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
                config=ConfigurationManager()
                model_trainer_config = config.get_model_trainer_config()
                model_trainer_config = ModelTrainer(config=model_trainer_config)
                model_trainer_config.train()