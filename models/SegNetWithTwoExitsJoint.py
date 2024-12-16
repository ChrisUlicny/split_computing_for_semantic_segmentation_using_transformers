from models import SegNetWithTwoExits as ees
from misc import ForwardStrategy
import torch
import metrics
from utils import utils
import os
import torch.nn as nn



class SegNetWithTwoExitsJoint(ees.SegNetWithTwoExits):

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 512]):
        super().__init__(in_channels, out_channels, features)
        self.dynamic_weights = False
        if self.dynamic_weights:
            self.costs = nn.Parameter(torch.tensor([0.4, 0.8, 0.7]))
            self.weights = torch.sigmoid(self.costs)
        else:
            self.weights = [0.45, 0.55, 0.7]

    def set_strategy(self, strategy: ForwardStrategy):
        if not isinstance(strategy, ForwardStrategy.ExitEnsemble) and not isinstance(strategy, ForwardStrategy.OnlineDistillation) \
            and not isinstance(strategy, ForwardStrategy.JointTrainingStrategy) and not isinstance(strategy, ForwardStrategy.AllInferenceStrategy) \
                and not isinstance(strategy, ForwardStrategy.InferenceStrategy):
            raise Exception("Invalid strategy for joint training")
        else:
            self.training_strategy = strategy

    def print_current_costs(self):
        print(f"first exit: {self.weights[0]}  second exit: {self.weights[1]}  final exit: {self.weights[2]}")


    def adjust_loss_weights(self, losses):
        losses = torch.tensor(losses)
        mean_loss = torch.mean(losses).item()
        # compute weights for exits based on their relative performance
        weights = torch.exp(-self.alpha * (losses/mean_loss))
        # normalize -> sum=1
        weights = weights/(weights.sum())
        self.costs = [i.item() for i in weights]

    def save_hyperparameters(self, hyperparameters: dict, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = f"{path}/hyperparameters.txt"
        hyperparameters["cost_first"] = self.weights[0]
        hyperparameters["cost_second"] = self.weights[1]
        hyperparameters["cost_final"] = self.weights[2]
        with open(full_path, 'w') as file:
            self.training_strategy.write_hyperparameters_to_file(hyperparameters, file)
            file.write('\n')

    def save_costs(self, array):
        if self.dynamic_weights:
            array.append([self.weights[0].item(), self.weights[1].item(), self.weights[2].item()])
        else:
            array.append([self.weights[0], self.weights[1], self.weights[2]])
        return array

    def check_batch_acc_miou_loss(self, loader,  loss_func=None):
        return metrics.check_batch_acc_miou_loss_joint(self, loader, self.num_classes, loss_func)

    def check_batch_miou(self, loader,  num_classes=None):
        return metrics.check_batch_miou_joint(self, loader, self.num_classes)

    def save_checkpoint(self, state, filename, folder, save_model, model_num=0):
        utils.save_checkpoint_joint(state, filename, folder, save_model, model_num)

    def calculate_loss(self, predictions, labels, loss_func, separate=False):
        return self.training_strategy.calculate_loss(predictions, labels, loss_func, separate)

    def append_metric_lists(self, metric_dict, epoch):
        val_accuracy = metric_dict.get("val_acc")
        val_loss = metric_dict.get("val_loss")
        train_loss = metric_dict.get("train_loss")
        train_accuracy = metric_dict.get("train_acc")
        val_miou = metric_dict.get("val_miou")
        train_miou = metric_dict.get("train_miou")
        val_acc_list = metric_dict.get("val_acc_list")
        train_acc_list = metric_dict.get("train_acc_list")
        val_miou_list = metric_dict.get("val_miou_list")
        train_miou_list = metric_dict.get("train_miou_list")

        val_acc_list.append(val_accuracy)
        train_acc_list.append(train_accuracy)
        val_miou_list.append(val_miou)
        train_miou_list.append(train_miou)

        metric_dict = {
            "val_acc_list": val_acc_list,
            "train_acc_list": train_acc_list,
            "val_miou_list": val_miou_list,
            "train_miou_list": train_miou_list,
            "val_acc": val_accuracy,
            "train_acc": train_accuracy,
            "val_miou": val_miou,
            "train_miou": train_miou,
            "val_loss": val_loss,
            "train_loss": train_loss
        }

        return metric_dict
