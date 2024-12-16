import json
from enum import Enum

import yaml
import TransformerTrainingStrategy


class OptimizerType(Enum):

    ADAM = "Adam"
    ADAMW = "AdamW"
    SGD = "SDG"


class ModelTypes(Enum):
    SEPARATE = "separate"
    JOINT = "joint"
    TRANSFORMER = "transformer"
    SWINLUNET = "SwinLUNET"
    SETR = "Setr"
    SWINUNET = "SwinUNET"
    SWINOFFICIAL = "SwinOfficial"

class TrainingConfig:
    def __init__(self):
        self.batch_size = 8
        # znnacne vacsie
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.model_name = "swinunet_citypretrained_frozen_all"
        self.train_model = True
        self.training_stages = [TransformerTrainingStrategy.TrainingBackbonePretrained()]
        self.inference_strategy = TransformerTrainingStrategy.TransformerInferenceStrategy()
        self.load_model = True
        self.model_to_load = "mask2former_swint_cityscapes.pkl"
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.make_prediction = True
        self.optimizer_type = OptimizerType.ADAMW
        self.repeat_model = 1
        self.gpu_num = 0
        self.training_type = ModelTypes.SWINUNET
        self.splits = None
        self.split_index = 0
        self.channels = [0, 3, 5, 5, 6, 5, 6, 3]
        self.reduction_factors = [0, 0, 0, 0, 0, 0, 0, 0]
        self.resuming_training = False
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def display(self):
        print("Batch Size:", self.batch_size)
        print("Learning Rate:", self.learning_rate)
        print("Number of Epochs:", self.num_epochs)
        print("Model Name:", self.model_name)
        print("Train Model:", self.train_model)
        print("Load Model:", self.load_model)
        print("Model to Load:", self.model_to_load)
        print("Make Prediction:", self.make_prediction)
        print("Optimizer type:", self.optimizer_type.value)
        if self.optimizer_type == OptimizerType.SGD:
            print("Momentum:", self.momentum)
            print("Weight decay:", self.weight_decay)
        print("Training Stages:", ", ".join([str(i.__name__) for i in self.training_stages]))
        print("Inference Strategy:", str(self.inference_strategy.__name__))
        print("Splits:", str(self.splits))
        print("Channels:", str(self.channels))
        print("Spatial reduction factors:", str(self.reduction_factors))
        print("Split index:", str(self.split_index))
        print("Resume training:", str(self.resuming_training))

    def save_to_json(self, filename):
        config_dict = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "train_model": self.train_model,
            "load_model": self.load_model,
            "model_to_load": self.model_to_load,
            "make_prediction": self.make_prediction,
            "optimizer_type": self.optimizer_type.value,
            "gpu_num": self.gpu_num,
            "training_type": self.training_type.value,
            "training_stages": ", ".join([str(type(i).__name__) for i in self.training_stages]),
            "inference_strategy": str(type(self.inference_strategy).__name__),
            "repeat_model": self.repeat_model,
            "splits": str(self.splits),
            "channels": self.channels,
            "reduction_factors": self.reduction_factors,
            "split_index": self.split_index,
            "resume_training": self.resuming_training
        }
        if self.optimizer_type == OptimizerType.SGD:
            config_dict["momentum"] = self.momentum
            config_dict["weight_decay"] = self.weight_decay

        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=4)

