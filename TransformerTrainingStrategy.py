import time

from torchvision.utils import save_image
from enum import Enum
import metrics
import utils.utils as utils
# from ForwardStrategy import *
import TrainingConfig
from torch import optim

import PoolingStrategy
import loss_functions
import torch.nn as nn
import confidence_calc
from utils.utils import *


class ExitPositionTransformer(Enum):
    FIRST_EXIT = 0
    BACKBONE = 1

class TransformerStrategy:

    def is_trained(self):
        raise NotImplementedError()

    def __init__(self):
        self.first_exit_position = ExitPositionTransformer.FIRST_EXIT.value
        self.final_exit_position = ExitPositionTransformer.BACKBONE.value

    def exit_first(self, network, inputs):
        raise NotImplementedError()

    def exit_final(self, network, inputs):
        raise NotImplementedError()

    def enter_first(self):
        raise NotImplementedError()

    def enter_final(self):
        raise NotImplementedError()

    def get_loss_func(self):
        raise NotImplementedError()

    def exit_split(self):
        return False

    def initialize_optimizer(self, network, optimizer_type, learning_rate, momentum):
        pass

    def calculate_loss(self, predictions, labels, loss_func, separate=False):
        loss = loss_func(predictions, labels)
        return loss

    def write_hyperparameters_to_file(self, hyperparameters, file):
        lr = hyperparameters.get("lr")
        bs = hyperparameters.get("bs")
        optimizer_type = hyperparameters.get("optimizer")
        momentum = hyperparameters.get("momentum")
        weight_decay = hyperparameters.get("weight_decay")
        file.write(f"Hyperparameters: start lr={lr}, batch size={str(bs)}, optimizer={optimizer_type}")
        if momentum is not None:
            file.write(f', momentum: {momentum}, weight_decay: {weight_decay}')

    def make_prediction(self, image, mask, model, idx, color_map, prediction_folder):
        raise TypeError("Can't make predictions with training strategy")

    def check_batch_acc_miou_loss(self, network, loader, num_classes, loss_func):
        return metrics.check_batch_acc_miou_loss(network, loader, num_classes, loss_func)



class TransformerTrainingExitOnly(TransformerStrategy):

    def is_trained(self):
        return True

    # def set_split_points(self, network):
    #     network.split_points = []
    #     network.exit_split_points = [True, True]

    def exit_first(self, network, inputs):
        return inputs["ee1"]

    def exit_final(self, network, inputs):
        return None

    def enter_first(self):
        return True

    def enter_final(self):
        return False


    def get_loss_func(self):
        return nn.CrossEntropyLoss()

    def initialize_optimizer(self, network, optimizer_type, learning_rate, momentum):
        layers_to_train = (list(network.stage1.parameters())
                           + list(network.stage2.parameters())
                           + list(network.decoder1.parameters())
                           + list(network.decoder2.parameters())
                           + list(network.split1.parameters())
                           + list(network.split2.parameters())
                           + list(network.exit_split1.parameters()))
        if optimizer_type == 'Adam':
            return optim.Adam(layers_to_train, lr=learning_rate)
        elif optimizer_type == 'SGD':
            return optim.SGD(layers_to_train, lr=learning_rate, momentum=momentum, weight_decay=0.0001)

    def calculate_loss(self, predictions, labels, loss_func, separate=False):
        loss = loss_func(predictions, labels)
        return loss



class TransformerExitInferenceStrategy(TransformerStrategy):

    def is_trained(self):
        return False


    def exit_first(self, network, inputs):
        return inputs["ee1"]

    def enter_first(self):
        return True

    def enter_final(self):
        return False


    def make_prediction(self, image, mask, model, idx, color_map, prediction_folder):
        if not os.path.exists(prediction_folder):
            # if the directory is not present, create it
            os.makedirs(prediction_folder)
        save_image(image, f"{prediction_folder}/original{idx}.png")
        mask = torch.argmax(mask, dim=3)
        mask = seg_map_to_image(mask, color_map)
        save_numpy_as_image(mask, f"{prediction_folder}/ground{idx}.png")
        prediction_for_accuracy = []
        start = time.time()
        output = model(image)
        end = time.time()
        prediction = output
        prediction = torch.squeeze(prediction)
        prediction_for_accuracy.append(prediction)
        # print('prediction size pre argmax after model:', prediction.shape)
        prediction = torch.argmax(prediction, dim=0)
        # print('prediction size:', prediction.shape)
        new_image = seg_map_to_image(prediction, color_map)
        # print('new image size:', new_image.shape)
        save_numpy_as_image(new_image, f"{prediction_folder}/pred{idx}.png")
        time_elapsed = end - start
        return prediction_for_accuracy, time_elapsed


class TransformerInferenceStrategy(TransformerStrategy):

    def is_trained(self):
        return False


    def exit_final(self, network, inputs):
        return inputs["backbone"]

    def enter_first(self):
        return False

    def enter_final(self):
        return True


    def make_prediction(self, image, mask, model, idx, color_map, prediction_folder):
        if not os.path.exists(prediction_folder):
            # if the directory is not present, create it
            os.makedirs(prediction_folder)
        save_image(image, f"{prediction_folder}/original{idx}.png")
        mask = torch.argmax(mask, dim=3)
        mask = seg_map_to_image(mask, color_map)
        save_numpy_as_image(mask, f"{prediction_folder}/ground{idx}.png")
        prediction_for_accuracy = []
        start = time.time()
        output = model(image)
        end = time.time()
        prediction = output
        prediction = torch.squeeze(prediction)
        prediction_for_accuracy.append(prediction)
        # print('prediction size pre argmax after model:', prediction.shape)
        prediction = torch.argmax(prediction, dim=0)
        # print('prediction size:', prediction.shape)
        new_image = seg_map_to_image(prediction, color_map)
        # print('new image size:', new_image.shape)
        save_numpy_as_image(new_image, f"{prediction_folder}/pred{idx}.png")
        time_elapsed = end - start
        return prediction_for_accuracy, time_elapsed


class TrainingBackbone(TransformerStrategy):

    def is_trained(self):
        return True

    # def set_split_points(self, network):
    #     network.split_points = []
    #     network.exit_split_points = [True, True]

    def exit_first(self, network, inputs):
        raise Exception("Got into first exit while training backbone")

    def exit_final(self, network, inputs):
        return inputs["backbone"]

    def enter_first(self):
        return False

    def enter_final(self):
        return True

    def get_loss_func(self):
        return nn.CrossEntropyLoss()

    def check_batch_acc_miou_no_loader(self, model, x, y, num_classes):
        num_correct = 0
        num_pixels = 0
        accuracy = 0
        miou = 0
        model.eval()

        with torch.no_grad():
            predictions = x
            miou_batch = 0
            predictions = torch.argmax(predictions, dim=1)
            y = torch.argmax(y, dim=3)
            num_correct += (predictions == y).sum()
            num_pixels += torch.numel(predictions)
            accuracy = accuracy + num_correct / num_pixels * 100
            bs = predictions.shape[0]
            for i in range(0, bs):
                miou_batch += metrics.calculate_mIoU(predictions[i], y[i], num_classes)

            miou_batch = miou_batch / bs
            miou += miou_batch

            predictions = predictions.to("cpu")
            y = y.to("cpu")

        model.train()

        # print("Average loss across all batches: " + str(round(avg_loss, 3)))
        return accuracy, miou

    def initialize_optimizer(self, network, learning_rate, optimizer_type, momentum):
        if optimizer_type == TrainingConfig.OptimizerType.ADAM:
            return optim.Adam(network.parameters(), lr=learning_rate)
        elif optimizer_type == TrainingConfig.OptimizerType.ADAMW:
            return optim.AdamW(network.parameters(), lr=learning_rate)
        elif optimizer_type == TrainingConfig.OptimizerType.SGD:
            return optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0001)
        else:
            raise ValueError(f"Unexpected optimizer type {optimizer_type}!")

    def calculate_loss(self, predictions, labels, loss_func, separate=False):
        loss = loss_func(predictions, labels)
        return loss

class TrainingBackbonePretrained(TrainingBackbone):

    def initialize_optimizer(self, network, learning_rate, optimizer_type, momentum):
        params_to_train = []
        for name, param in network.named_parameters():
            if 'blocks' not in name and 'downsample' not in name and "patch_embed" not in name:
            # if 'layers.' in name and 'blocks' in name or 'patch_embed' in name:
                print("Training:", name)
                params_to_train.append(param)
        if not params_to_train:
            raise ValueError("No parameters to train found.")

        if optimizer_type == TrainingConfig.OptimizerType.ADAM:
            return optim.Adam(params_to_train, lr=learning_rate)
        elif optimizer_type == TrainingConfig.OptimizerType.ADAMW:
            return optim.AdamW(params_to_train, lr=learning_rate)
        elif optimizer_type == TrainingConfig.OptimizerType.SGD:
            return optim.SGD(params_to_train, lr=learning_rate, momentum=momentum, weight_decay=0.0001)
        else:
            raise ValueError(f"Unexpected optimizer type {optimizer_type}!")

class TrainingWholeNetworkWithSplit(TrainingBackbone):
    def initialize_optimizer(self, network, learning_rate, optimizer_type, momentum):
        params_to_train = []
        for name, param in network.named_parameters():
            if 'split' in name:
                print("Training:", name)
                params_to_train.append(param)
        if not params_to_train:
            raise ValueError("No parameters with 'split' in their names found.")

        if optimizer_type == TrainingConfig.OptimizerType.ADAM:
            return optim.Adam(params_to_train, lr=learning_rate)
        elif optimizer_type == TrainingConfig.OptimizerType.SGD:
            return optim.SGD(params_to_train, lr=learning_rate, momentum=momentum, weight_decay=0.0001)
        else:
            raise ValueError(f"Unexpected optimizer type {optimizer_type}!")


class TrainingSplitsOneByOne(TrainingBackbone):

    def exit_final(self, network, inputs):
        return [inputs["backbone"], inputs["after_split_output"], inputs["before_split_label"]]

    def exit_split(self):
        return False

    def get_loss_func(self):
        return nn.MSELoss()

    def calculate_loss(self, predictions, labels, loss_func, separate=False):
        loss = loss_func(predictions[1], predictions[2])
        return loss

    def check_batch_acc_miou_loss(self, model, loader, num_classes, loss_func):
        num_correct = 0
        num_pixels = 0
        accuracy = 0
        counter = 0
        total_loss = 0
        miou = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=utils.get_device())
                y = y.to(device=utils.get_device())
                predictions = model(x)
                loss = model.calculate_loss(predictions, None, loss_func)
                predictions = predictions[0]
                miou_batch = 0
                total_loss = total_loss + loss
                predictions = torch.argmax(predictions, dim=1)
                y = torch.argmax(y, dim=3)
                num_correct += (predictions == y).sum()
                num_pixels += torch.numel(predictions)
                accuracy = accuracy + num_correct / num_pixels * 100
                bs = predictions.shape[0]
                for i in range(0, bs):
                    miou_batch += metrics.calculate_mIoU(predictions[i], y[i], num_classes)

                miou_batch = miou_batch / bs
                miou += miou_batch

                counter += 1
                x = x.to("cpu")
                y = y.to("cpu")

        model.train()

        avg_batch_acc = (accuracy / counter).item()
        # print("Average accuracy across all batches: " + str(round(avg_batch_acc, 2)) + "%")

        avg_batch_miou = (miou / counter).item() * 100
        # print("Average mIoU across all batches: " + str(round(avg_batch_miou, 2)) + "%")

        avg_loss = (total_loss / counter).item()
        # print("Average loss across all batches: " + str(round(avg_loss, 3)))
        return [avg_batch_acc], [avg_batch_miou], [avg_loss]


    def check_batch_acc_miou_no_loader(self, model, x, y, num_classes):
        num_correct = 0
        num_pixels = 0
        accuracy = 0
        miou = 0
        model.eval()

        with torch.no_grad():
            predictions = x
            predictions = predictions[0]
            miou_batch = 0
            predictions = torch.argmax(predictions, dim=1)
            y = torch.argmax(y, dim=3)
            num_correct += (predictions == y).sum()
            num_pixels += torch.numel(predictions)
            accuracy = accuracy + num_correct / num_pixels * 100
            bs = predictions.shape[0]
            for i in range(0, bs):
                miou_batch += metrics.calculate_mIoU(predictions[i], y[i], num_classes)

            miou_batch = miou_batch / bs
            miou += miou_batch

            predictions = predictions.to("cpu")
            y = y.to("cpu")

        model.train()

        # print("Average loss across all batches: " + str(round(avg_loss, 3)))
        return accuracy, miou

    def initialize_optimizer(self, network, learning_rate, optimizer_type, momentum):
        params_to_train = []
        for name, param in network.named_parameters():
            # print(name)
            if 'spatial' in name or 'channel' in name:
                print("Training:", name)
                params_to_train.append(param)

        # exit('Finished network iteration when initializing optimizer')
        if not params_to_train:
            raise ValueError("No split parameters to train found.")

        if optimizer_type == TrainingConfig.OptimizerType.ADAM:
            return optim.Adam(params_to_train, lr=learning_rate)
        elif optimizer_type == TrainingConfig.OptimizerType.SGD:
            return optim.SGD(params_to_train, lr=learning_rate, momentum=momentum, weight_decay=0.0001)
        elif optimizer_type == TrainingConfig.OptimizerType.ADAMW:
            return optim.AdamW(params_to_train, lr=learning_rate)
        else:
            raise ValueError(f"Unexpected optimizer type {optimizer_type}!")



class CreateSplitDataset(TransformerStrategy):
    def is_trained(self):
        return False

    def exit_final(self, network, inputs):
        raise ValueError("Should not reach final exit")

    def enter_final(self):
        return True

    def exit_split(self):
        return True


class TrainSplitOnly(TrainingSplitsOneByOne):
    def is_trained(self):
        return True

    def calculate_loss(self, predictions, labels, loss_func, separate=False):
        loss = loss_func(predictions, labels)
        return loss


    def get_loss_func(self):
        return nn.MSELoss()

    def initialize_optimizer(self, network, learning_rate, optimizer_type, momentum):

        if optimizer_type == TrainingConfig.OptimizerType.ADAM:
            return optim.Adam(network.parameters(), lr=learning_rate)
        elif optimizer_type == TrainingConfig.OptimizerType.SGD:
            return optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0001)
        elif optimizer_type == TrainingConfig.OptimizerType.ADAMW:
            return optim.AdamW(network.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unexpected optimizer type {optimizer_type}!")


    def check_batch_loss(self, model, loader, loss_func):
        counter = 0
        total_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=utils.get_device())
                y = y.to(device=utils.get_device())
                predictions = model(x)
                loss = model.calculate_loss(predictions, y, loss_func)
                total_loss = total_loss + loss
                counter += 1
                x = x.to("cpu")
                y = y.to("cpu")

        model.train()
        avg_loss = (total_loss / counter).item()
        # print("Average loss across all batches: " + str(round(avg_loss, 3)))
        return [avg_loss]


    def check_batch_acc_miou_no_loader(self, model, x, y, num_classes):
        num_correct = 0
        num_pixels = 0
        accuracy = 0
        miou = 0
        model.eval()

        with torch.no_grad():
            predictions = x
            predictions = predictions[0]
            miou_batch = 0
            predictions = torch.argmax(predictions, dim=1)
            y = torch.argmax(y, dim=3)
            num_correct += (predictions == y).sum()
            num_pixels += torch.numel(predictions)
            accuracy = accuracy + num_correct / num_pixels * 100
            bs = predictions.shape[0]
            for i in range(0, bs):
                miou_batch += metrics.calculate_mIoU(predictions[i], y[i], num_classes)

            miou_batch = miou_batch / bs
            miou += miou_batch

            predictions = predictions.to("cpu")
            y = y.to("cpu")

        model.train()

        # print("Average loss across all batches: " + str(round(avg_loss, 3)))
        return accuracy, miou



