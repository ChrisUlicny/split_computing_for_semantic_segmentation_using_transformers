from enum import Enum

from torch import optim

import PoolingStrategy
import loss_functions
import torch.nn as nn
import confidence_calc
from utils.utils import *



class ExitPosition(Enum):

    FIRST_EXIT = 0
    SECOND_EXIT = 1
    BACKBONE = 2


class GeneralStrategy:
    def is_trained(self):
        raise NotImplementedError()

class ForwardStrategy(GeneralStrategy):

    def __init__(self):
        self.first_exit_position = ExitPosition.FIRST_EXIT.value
        self.second_exit_position = ExitPosition.SECOND_EXIT.value
        self.final_exit_position = ExitPosition.BACKBONE.value

    def exit_first(self, network, inputs):
        raise NotImplementedError()

    def exit_second(self, network, inputs):
        raise NotImplementedError()

    def exit_final(self, network, inputs):
        raise NotImplementedError()

    def enter_first(self):
        raise NotImplementedError()

    def enter_second(self):
        raise NotImplementedError()

    def enter_final(self):
        raise NotImplementedError()

    def get_loss_func(self):
        raise NotImplementedError()

    def initialize_optimizer(self, network, optimizer_type, learning_rate, momentum):
        raise NotImplementedError()

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

    def is_trained(self):
        raise NotImplementedError()



class TrainingBackboneStrategy(ForwardStrategy):

    def exit_first(self, network, inputs):
        raise Exception("Got into first exit while training the final one")

    def exit_second(self, network, inputs):
        raise Exception("Got into second exit while training the final one")

    def exit_final(self, network, inputs):
        if len(inputs) > 1:
            return inputs[self.final_exit_position]
        else:
            return inputs

    def enter_first(self):
        return False

    def enter_second(self):
        return False

    def enter_final(self):
        return True

    def is_trained(self):
        return True

    def get_loss_func(self):
        return nn.CrossEntropyLoss()

    def initialize_optimizer(self, network, optimizer_type, learning_rate, momentum):
        print("Training backbone", network.pooling_strategy.has_parameters())
        if network.pooling_strategy.has_parameters():
            pooling_layers_parameters = list(network.pooling_convs) + list(network.unpooling_convs)
            if optimizer_type == 'Adam':
                return optim.Adam(list(network.downs.parameters()) + list(network.ups.parameters()) +
                                  list(network.final_conv.parameters()) + list(pooling_layers_parameters),
                                  lr=learning_rate)
            elif optimizer_type == 'SGD':
                return optim.SGD(list(network.downs.parameters()) + list(network.ups.parameters()) +
                                 list(network.final_conv.parameters()) + list(pooling_layers_parameters),
                                 lr=learning_rate, momentum=momentum, weight_decay=0.0001)
        else:
            if optimizer_type == 'Adam':
                return optim.Adam(list(network.downs.parameters()) + list(network.ups.parameters()) +
                                  list(network.final_conv.parameters()), lr=learning_rate)
            elif optimizer_type == 'SGD':
                return optim.SGD(list(network.downs.parameters()) + list(network.ups.parameters()) +
                                  list(network.final_conv.parameters()), lr=learning_rate, momentum=momentum, weight_decay=0.0001)


class InferenceStrategy(ForwardStrategy):

    def exit_first(self, network, inputs):
        print("First exit:")
        confidence, _ = confidence_calc.calculate_confidence(inputs[self.first_exit_position])
        print("Confidence: " + str(round(confidence, 2)) + "%")
        if network.thresholds[self.first_exit_position] < confidence:
            network.exit_counts[self.first_exit_position] += 1
            print("Exiting through first exit")
            return inputs[self.first_exit_position], confidence, 0
        else:
            return None

    def exit_second(self, network, inputs):
        print("Second exit:")
        confidence, _ = confidence_calc.calculate_confidence(inputs[self.second_exit_position])
        print("Confidence: " + str(round(confidence, 2)) + "%")
        if network.thresholds[self.second_exit_position] < confidence:
            print("Exiting through second exit")
            network.exit_counts[self.second_exit_position] += 1
            return inputs[self.second_exit_position], confidence, 1
        else:
            return None

    def exit_final(self, network, inputs):
        print("Final exit:")
        confidence, _ = confidence_calc.calculate_confidence(inputs[self.final_exit_position])
        network.exit_counts[self.final_exit_position] += 1
        print("Exiting through final exit")
        return inputs[self.final_exit_position], confidence, 2

    def enter_first(self):
        return True

    def enter_second(self):
        return True

    def enter_final(self):
        return True

    def get_loss_func(self):
        pass

    def initialize_optimizer(self, network, optimizer_type, learning_rate, momentum):
        pass

    def calculate_loss(self, predictions, labels, loss_func, separate=False):
        pass

    def write_hyperparameters_to_file(self, hyperparameters, file):
        pass

    def is_trained(self):
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
        prediction, _, exit_num = output
        exit_names = ['first', 'second', 'final']
        prediction = torch.squeeze(prediction)
        _, confidence_map = confidence_calc.calculate_confidence(prediction)
        generate_confidence_map(confidence_map,
                                f"{prediction_folder}/confidence_map{idx}_{exit_names[exit_num]}.png")
        prediction_for_accuracy.append(prediction)
        # print('prediction size pre argmax after model:', prediction.shape)
        prediction = torch.argmax(prediction, dim=0)
        # print('prediction size:', prediction.shape)
        new_image = seg_map_to_image(prediction, color_map)
        # print('new image size:', new_image.shape)
        save_numpy_as_image(new_image, f"{prediction_folder}/pred{idx}_{exit_names[exit_num]}.png")
        time_elapsed = end - start
        return prediction_for_accuracy, time_elapsed


class EnsembleInference(InferenceStrategy):

    def exit_second(self, network, inputs):
        print("Second exit:")
        ensemble = (inputs[self.second_exit_position] + inputs[self.first_exit_position])/2
        # ensemble = inputs[self.second_exit_position]
        confidence, _ = confidence_calc.calculate_confidence(ensemble)
        print("Confidence: " + str(round(confidence, 2)) + "%")
        if network.thresholds[self.second_exit_position] < confidence:
            print("Exiting through second exit")
            network.exit_counts[self.second_exit_position] += 1
            return ensemble, confidence, 1
        else:
            return None

    def exit_final(self, network, inputs):
        print("Final Exit:")
        print("Exiting through final exit")
        ensemble = sum(inputs)/3
        # ensemble = (inputs[self.second_exit_position] + inputs[self.final_exit_position]) / 2
        confidence, _ = confidence_calc.calculate_confidence(ensemble)
        network.exit_counts[self.final_exit_position] += 1
        return ensemble, confidence, 2



class JointTrainingStrategy(ForwardStrategy):

    def exit_first(self, network, inputs):
        return None

    def exit_second(self, network, inputs):
        return None

    def exit_final(self, network, inputs):
        first_tuple = (inputs[self.first_exit_position], network.weights[self.first_exit_position])
        second_tuple = (inputs[self.second_exit_position], network.weights[self.second_exit_position])
        final_tuple = (inputs[self.final_exit_position], network.weights[self.final_exit_position])
        return final_tuple, second_tuple, first_tuple

    def enter_first(self):
        return True

    def enter_second(self):
        return True

    def enter_final(self):
        return True

    def get_loss_func(self):
        return loss_functions.loss_func_joint_training

    def calculate_loss(self, predictions, labels, loss_func, separate=False):
        return loss_func(predictions, labels, separate)

    def is_trained(self):
        return True

    def initialize_optimizer(self, network, optimizer_type, learning_rate, momentum):
        print("Joint training")
        if optimizer_type == 'Adam':
            return optim.Adam(network.parameters(), lr=learning_rate)
            # optimizer = optim.Adam([{"params": self.ups.parameters()},
            #                         {'params': self.downs.parameters()},
            #                         {'params': self.exit_ups.parameters()},
            #                         {'params': self.first_exit_ups.parameters()},
            #                         {'params': self.final_conv.parameters()},
            #                         {'params': self.final_exit_conv.parameters()}]
            #                        # {'params': self.costs, 'lr': 0.01}]
            #                         ,lr=learning_rate)
        elif optimizer_type == 'SGD':
            # optimizer = optim.SGD([{"params": network.ups.parameters()},
            #                         {'params': network.downs.parameters()},
            #                         {'params': network.exit_ups.parameters()},
            #                         {'params': network.first_exit_ups.parameters()},
            #                         {'params': network.final_conv.parameters()},
            #                         {'params': network.final_exit_conv.parameters()},
            #                         {'params': network.costs, 'lr': 0.01}]
            #                          ,lr=learning_rate, momentum=momentum)
            optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0001)
            return optimizer

    def write_hyperparameters_to_file(self, hyperparameters: dict, file):
        lr = hyperparameters.get("lr")
        bs = hyperparameters.get("bs")
        optimizer_type = hyperparameters.get("optimizer")
        momentum = hyperparameters.get("momentum")
        weight_decay = hyperparameters.get("weight_decay")
        cost1 = hyperparameters.get("cost_first")
        cost2 = hyperparameters.get("cost_second")
        cost3 = hyperparameters.get("cost_final")
        file.write(f"Hyperparameters: start lr={lr}, batch size={str(bs)}, optimizer={optimizer_type}")
        file.write(f" cost first = {cost1}, cost second = {cost2}, cost final = {cost3}")
        if momentum is not None:
            file.write(f'momentum = {momentum}, weight_decay = {weight_decay}')


class AllInferenceStrategy(InferenceStrategy):

    def exit_first(self, network, inputs):
        return None

    def exit_second(self, network, inputs):
        return None

    def exit_final(self, network, inputs):
        # ensemble = [inputs[self.first_exit_position],
        #             (inputs[self.second_exit_position]+inputs[self.first_exit_position])/2,
        #             (sum(inputs))/3]
        return inputs

    def enter_first(self):
        return True

    def enter_second(self):
        return True

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

        exit_names = ['first', 'second', 'final']
        for i, prediction in enumerate(output):
            # print(f"{exit_names[i]} exit:")
            prediction = torch.squeeze(prediction)
            _, confidence_map = confidence_calc.calculate_confidence(prediction)
            generate_confidence_map(confidence_map,
                                    f"{prediction_folder}/confidence_map{idx}_{exit_names[i]}.png")
            prediction_for_accuracy.append(prediction)
            prediction = torch.argmax(prediction, dim=0)
            new_image = seg_map_to_image(prediction, color_map)
            save_numpy_as_image(new_image, f"{prediction_folder}/pred{idx}_{exit_names[i]}.png")
        time_elapsed = end - start
        return prediction_for_accuracy, time_elapsed



class ExitEnsemble(JointTrainingStrategy):

    def __init__(self, hyperparameters: dict):
        super().__init__()
        self.alpha = hyperparameters.get('alpha')
        self.gamma = hyperparameters.get('gamma')
        self.lamda = hyperparameters.get('lambda')

    def get_loss_func(self):
        return loss_functions.loss_func_exit_ensemble

    def calculate_loss(self, predictions, labels, loss_func, separate=False):
        return loss_func(predictions, labels, self.alpha, self.gamma, self.lamda, separate)

    def write_hyperparameters_to_file(self, hyperparameters: dict, file):
        lr = hyperparameters.get("lr")
        bs = hyperparameters.get("bs")
        optimizer_type = hyperparameters.get("optimizer")
        momentum = hyperparameters.get("momentum")
        weight_decay = hyperparameters.get("weight_decay")
        cost1 = hyperparameters.get("cost_first")
        cost2 = hyperparameters.get("cost_second")
        cost3 = hyperparameters.get("cost_final")
        alpha = hyperparameters.get("alpha")
        gamma = hyperparameters.get("gamma")
        lamda = hyperparameters.get("lambda")
        file.write(f"Hyperparameters: start lr={lr}, batch size={str(bs)}, optimizer={optimizer_type}")
        file.write(f" cost first = {cost1}, cost second = {cost2}, cost final = {cost3}")
        file.write(f" alpha = {alpha}, gamma = {gamma}, lambda = {lamda}")
        if momentum is not None:
            file.write(f'momentum = {momentum}, weight_decay = {weight_decay}')


class OnlineDistillation(JointTrainingStrategy):

    def __init__(self, hyperparameters: dict):
        super().__init__()
        self.alpha = hyperparameters.get('alpha')

    def get_loss_func(self):
        return loss_functions.loss_func_joint_distillation

    def write_hyperparameters_to_file(self, hyperparameters: dict, file):
        lr = hyperparameters.get("lr")
        bs = hyperparameters.get("bs")
        optimizer_type = hyperparameters.get("optimizer")
        momentum = hyperparameters.get("momentum")
        weight_decay = hyperparameters.get("weight_decay")
        cost1 = hyperparameters.get("cost_first")
        cost2 = hyperparameters.get("cost_second")
        cost3 = hyperparameters.get("cost_final")
        alpha = hyperparameters.get("alpha")
        file.write(f"Hyperparameters: start lr={lr}, batch size={str(bs)}, optimizer={optimizer_type}")
        file.write(f" cost first = {cost1}, cost second = {cost2}, cost final = {cost3},  alpha = {alpha}")
        if momentum is not None:
            file.write(f'momentum = {momentum}, weight_decay = {weight_decay}')

    def calculate_loss(self, predictions, labels, loss_func, separate=False):
        return loss_func(predictions, labels, self.alpha, separate)


class TrainingEarlyExits(ForwardStrategy):

    def exit_first(self, network, inputs):
        return None

    def exit_second(self, network, inputs):
        return [inputs[self.first_exit_position], inputs[self.second_exit_position]]

    def exit_final(self, network, inputs):
        raise Exception("Got into final exit while training the early exits")

    def enter_first(self):
        return True

    def enter_second(self):
        return True

    def enter_final(self):
        return False

    def is_trained(self):
        return True

    def get_loss_func(self):
        return loss_functions.loss_func_exits

    def initialize_optimizer(self, network, optimizer_type, learning_rate, momentum):
        if network.pooling_strategy.has_parameters:
            pooling_layers_parameters = list(network.pooling_convs) + list(network.unpooling_convs)
        else:
            pooling_layers_parameters = None
        print("Training early exits")
        if optimizer_type == 'Adam':
            return optim.Adam(list(network.exit_ups.parameters()) +
                              list(network.first_exit_ups.parameters()) +
                              list(network.final_exit_conv.parameters() + pooling_layers_parameters), lr=learning_rate)
        elif optimizer_type == 'SGD':
            return optim.SGD(list(network.exit_ups.parameters()) +
                             list(network.first_exit_ups.parameters()) +
                             list(network.final_exit_conv.parameters() + pooling_layers_parameters), lr=learning_rate, momentum=momentum, weight_decay=0.0001)


class DistillationEarlyExitsFromFinal(TrainingEarlyExits):

    def __init__(self, hyperparameters: dict):
        super().__init__()
        self.alpha = hyperparameters.get("alpha")

    def exit_second(self, network, inputs):
        return None

    def exit_final(self, network, inputs):
        return inputs

    def enter_final(self):
        return True

    def get_loss_func(self):
        return loss_functions.dist_exits_loss_func

    def calculate_loss(self, predictions, labels, loss_func, separate=False):
        return loss_func(predictions, labels, self.alpha)

    def write_hyperparameters_to_file(self, hyperparameters: dict, file):
        lr = hyperparameters.get("lr")
        bs = hyperparameters.get("bs")
        optimizer_type = hyperparameters.get("optimizer")
        momentum = hyperparameters.get("momentum")
        weight_decay = hyperparameters.get("weight_decay")
        alpha = hyperparameters.get("alpha")
        file.write(f"Hyperparameters: start lr={lr}, batch size={str(bs)}, optimizer={optimizer_type}, alpha = {alpha}")
        if momentum is not None:
            file.write(f'momentum = {momentum}, weight_decay = {weight_decay}')
