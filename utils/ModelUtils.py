import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.graph_utils import get_used_memory
import utils.utils as utils
import metrics
import time
import os
import pickle


class GeneralModel:

    def save_checkpoint(self, state, model_name, folder, save_model, model_num=0):
        raise NotImplementedError()
        # print("Saving checkpoint")
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # if not os.path.exists(f"{folder}/{model_name}_{str(model_num)}"):
        #     os.makedirs(f"{folder}/{model_name}_{str(model_num)}")
        # filename = f"{folder}/{model_name}_{str(model_num)}/{model_name}.pth.tar"
        # logname = f"{folder}/{model_name}_{str(model_num)}/{model_name}.txt"
        # torch.save(state, filename)
        # with open(logname, 'a') as file:
        #     file.write("epoch: ")
        #     file.write(str(state["epoch"]))
        #     file.write(" training loss: ")
        #     file.write(str(round(state["train_loss"], 3)))
        #     file.write(" training accuracy: ")
        #     file.write(str(round(state["train_accuracy"], 2)))
        #     file.write(" training mIoU: ")
        #     file.write(str(round(state["train_miou"], 2)))
        #     file.write(" validation loss: ")
        #     file.write(str(round(state["val_loss"], 3)))
        #     file.write(" validation accuracy: ")
        #     file.write(str(round(state["val_accuracy"], 2)))
        #     file.write(" validation mIoU: ")
        #     file.write(str(round(state["val_miou"], 2)))
        #     file.write('\n')

    def load_checkpoint(self, path):
        raise NotImplementedError()

    def check_batch_acc_miou_no_loader(self, x, y):
        raise NotImplementedError()

    def initialize_optimizer(self, learning_rate, optimizer_type):
        raise NotImplementedError()

    def check_batch_acc_miou_loss(self, loader, loss_func):
        raise NotImplementedError()

    def check_batch_miou(self, loader, num_classes):
        raise NotImplementedError()

    def get_loss_func(self):
        raise NotImplementedError()

    def save_hyperparameters(self, hyperparameters, path):
        raise NotImplementedError()

    def adjust_loss_weights(self, losses):
        pass

    def print_current_costs(self):
        pass

    def save_costs(self, array):
        pass

    def calculate_loss(self, predictions, labels, loss_func):
        return loss_func(predictions, labels)

    def append_metric_lists(self, list_dict, epoch):
        raise NotImplementedError()

    def draw_losses_graph_joint(self, train_acc_list, val_acc_list, file, metric_type, exits_num=3):
        raise NotImplementedError()


class GeneralModel2(nn.Module):
    def __init__(self):
        super(GeneralModel2, self).__init__()
        self.strategy = None
        self.results = {
            "ee1": None,
            "backbone": None,
            "before_split_label": None,
            "after_split_output": None,
            "transferred_datasize": [0]
        }
        self.datasizes = []
        self.num_classes = None

    def initialize_optimizer(self, learning_rate, optimizer_type, momentum=0.95):
        return self.strategy.initialize_optimizer(self, learning_rate, optimizer_type, momentum)

    def save_checkpoint(self, state, filename, folder, save_model, model_num=0):
        utils.save_checkpoint_split_json(state, filepath=f"{folder}/{filename}", save_model=save_model, model_num=model_num, exits_num=1)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.load_state_dict(state_dict=checkpoint["state_dict"], strict=False)
        return checkpoint


    def load_pretrained_checkpoint(self, path):

        # Load the checkpoint from the .pkl file
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        # Assuming the checkpoint contains a state dictionary
        # If the state dict is stored under a specific key, e.g., 'state_dict', access it accordingly
        # if 'state_dict' in checkpoint:
        #     state_dict = checkpoint['state_dict']
        # else:
        #     state_dict = checkpoint
        #
        # # Print the keys in the loaded state dictionary
        # print("Keys in the loaded state dictionary:")
        pretrained_state_dict = checkpoint['model']
        # for key, value in pretrained_state_dict.items():
        #     value = torch.tensor(value)
        #     pretrained_state_dict.value = value
        #     print(f"{key}: {value.shape}")
        # checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


        # layer_mapping = {
        #     'conv1.weight': 'conv1.weight',
        #     'conv1.bias': 'conv1.bias',
        #     'conv2.weight': 'conv2.weight',
        #     'conv2.bias': 'conv2.bias',
        #     'fc.weight': 'fc.weight',
        #     'fc.bias': 'fc.bias'
        # }

        # # Create a new state_dict for modelB using the mapping
        # new_state_dict = {}
        # for modelA_name, modelB_name in layer_mapping.items():
        #     print('Assigning {} to {}'.format(modelA_name, modelB_name))
        #     new_state_dict[modelB_name] = pretrained_state_dict[modelA_name]


        # extended_state_dict = pretrained_state_dict.copy()
        # for name, param in pretrained_state_dict.items():
        #     if "layers.2.blocks" in name:
        #         extended_state_dict[name.replace("layers.2.blocks", "layers_up.1.blocks")] = torch.tensor(param).clone()
        #     elif "layers.1.blocks" in name:
        #         extended_state_dict[name.replace("layers.1.blocks", "layers_up.2.blocks")] = torch.tensor(param).clone()
        #     elif "layers.0.blocks" in name:
        #         extended_state_dict[name.replace("layers.0.blocks", "layers_up.3.blocks")] = torch.tensor(param).clone()

        # for pretrained mask2former
        new_state_dict = {}
        for name, param in pretrained_state_dict.items():
            if "backbone.layers.2.blocks" in name:
                new_state_dict[name.replace("backbone.layers.2.blocks", "layers_up.1.blocks")] = torch.tensor(param).clone()
                new_state_dict[name.replace("backbone.layers.2.blocks", "layers.2.blocks")] = torch.tensor(param).clone()
            elif "backbone.layers.1.blocks" in name:
                new_state_dict[name.replace("backbone.layers.1.blocks", "layers_up.2.blocks")] = torch.tensor(param).clone()
                new_state_dict[name.replace("backbone.layers.1.blocks", "layers.1.blocks")] = torch.tensor(param).clone()
            elif "backbone.layers.0.blocks" in name:
                new_state_dict[name.replace("backbone.layers.0.blocks", "layers_up.3.blocks")] = torch.tensor(param).clone()
                new_state_dict[name.replace("backbone.layers.0.blocks", "layers.0.blocks")] = torch.tensor(
                    param).clone()
            elif "backbone.layers.3.blocks" in name:
                new_state_dict[name.replace("backbone.layers.3.blocks", "layers.3.blocks")] = torch.tensor(
                    param).clone()



        # print("\nPretrained state dict keys and shapes:")
        # for key, value in extended_state_dict.items():
        #     print(f"{key}: {value.shape}")
        # exit('Exited GeneralModel2 loading prerained checkpoint')


        # Load the new state_dict into modelB
        self.load_state_dict(state_dict=new_state_dict, strict=False)
        # compare_state_dicts(self.state_dict(), new_state_dict)
        #     # create mapping
        # mapping = [['fc1', 'fc1']]
        # for m in mapping:
        #     print('loading {} to {}'.format(m[1], m[0]))
        #     getattr(self, m[0]).load_state_dict(getattr(model, m[1]).state_dict())
        return None



    def check_batch_acc_miou_loss(self, loader, loss_func):
        return self.strategy.check_batch_acc_miou_loss(self, loader, self.num_classes, loss_func)

    def check_batch_acc_miou_loss_no_loader(self, x, y):
        return self.strategy.check_batch_acc_miou_no_loader(self, x, y, self.num_classes)

    def check_batch_miou(self, loader, num_classes=None):
        return metrics.check_batch_miou(self, loader, self.num_classes)

    def get_loss_func(self):
        return self.strategy.get_loss_func()

    def calculate_loss(self, predictions, labels, loss_func):
        return self.strategy.calculate_loss(predictions, labels, loss_func)

    def set_strategy(self, strategy):
        self.strategy = strategy

    def save_hyperparameters(self, hyperparameters: dict, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = f"{path}/hyperparameters.txt"
        with open(full_path, 'w') as file:
            lr = hyperparameters.get("lr")
            bs = hyperparameters.get("bs")
            optimizer_type = hyperparameters.get("optimizer")
            momentum = hyperparameters.get("momentum")
            weight_decay = hyperparameters.get("weight_decay")
            file.write(f"Hyperparameters: start lr={lr}, batch size={bs}, optimizer={optimizer_type}")
            if momentum is not None:
                file.write(f', momentum: {momentum}, weight_decay: {weight_decay}')
            file.write('\n')

    def reset_exit_counts(self):
        pass

    def draw_losses_graph_joint(self, train_acc_list, val_acc_list, file, metric_type, exits_num=None):
        utils.draw_losses_graph_backbone(train_acc_list, val_acc_list, file, metric_type)

    def get_split_point(self):
        return self.split_points

    def make_prediction(self, image, mask, model, idx, color_map, prediction_folder):
        if not os.path.exists(prediction_folder):
            os.makedirs(prediction_folder)
        utils.save_image(image, f"{prediction_folder}/original{idx}.png")
        mask = torch.argmax(mask, dim=3)
        mask = utils.seg_map_to_image(mask, color_map)
        utils.save_numpy_as_image(mask, f"{prediction_folder}/ground{idx}.png")
        prediction_for_accuracy = []
        start = time.time()
        output = model(image)
        end = time.time()
        prediction = output
        prediction = torch.squeeze(prediction)
        prediction_for_accuracy.append(prediction)
        prediction = torch.argmax(prediction, dim=0)
        new_image = utils.seg_map_to_image(prediction, color_map)
        utils.save_numpy_as_image(new_image, f"{prediction_folder}/pred{idx}.png")
        time_elapsed = end - start
        return prediction_for_accuracy, time_elapsed

    def get_datasizes(self, mb):
        if mb:
            self.datasizes = [i / 1048576 for i in self.results["transferred_datasize"]]
            return self.datasizes
        else:
            return [self.results["transferred_datasize"]]

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
        train_loss_list = metric_dict.get("train_loss_list")
        val_loss_list = metric_dict.get("val_loss_list")

        if len(val_acc_list) > epoch:
            val_acc_list[epoch][0] = val_accuracy[0]
            val_miou_list[epoch][0] = val_miou[0]
            val_loss_list[epoch][0] = val_loss[0]
            train_acc_list[epoch][0] = train_accuracy[0]
            train_miou_list[epoch][0] = train_miou[0]
            train_loss_list[epoch][0] = train_loss[0]
        else:
            val_miou_list.append(val_miou)
            val_acc_list.append(val_accuracy)
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)
            train_miou_list.append(train_miou)
            train_acc_list.append(train_accuracy)

        metric_dict = {
            "val_acc_list": val_acc_list,
            "train_acc_list": train_acc_list,
            "val_miou_list": val_miou_list,
            "train_miou_list": train_miou_list,
            "val_loss_list": val_loss_list,
            "train_loss_list": train_loss_list,
            "val_acc": val_accuracy,
            "train_acc": train_accuracy,
            "val_miou": val_miou,
            "train_miou": train_miou,
            "val_loss": val_loss,
            "train_loss": train_loss
        }

        return metric_dict

    def save_costs(self, array):
        pass


def compare_state_dicts(original_dict, extended_dict):
    for original_key, original_param in original_dict.items():


        if original_key not in extended_dict.keys():
            print(f"{original_key} not in pretrained layers")
        elif not torch.equal(original_param, extended_dict[original_key]):
            print(f"Mismatch found in {original_key}")
        else:
            print(f"{original_key} match correctly.")

class SingleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # padding = 1 which makes it same convolution (same input and output size)
            # bias is False because we are going to use batch norm
            # bias is going to be cancelled by batch norm
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # inplace True saves memory during both training and testing
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TransposeConv(nn.Module):
    # def __init__(self, hidden_dim, kernel_size=4, stride=2, padding=1):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2), stride=2, padding=0):
        super(TransposeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AvgUnpoolingLayer(nn.Module):
    def __init__(self):
        super(AvgUnpoolingLayer, self).__init__()

    def forward(self, inputs, output_size):
        unpooled_output = F.interpolate(inputs, size=output_size[-2:], mode='nearest')
        return unpooled_output


class SpatialReductionUnit(nn.Module):
    def __init__(self, in_channels, reduction_factor):
        super(SpatialReductionUnit, self).__init__()
        self.reduction_factor = reduction_factor
        # Calculate padding to maintain output size
        # padding = reduction_factor // 2 if reduction_factor % 2 == 1 else (reduction_factor - 1) // 2
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(reduction_factor, reduction_factor),
                              stride=reduction_factor)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x



class SpatialRestorationUnit(nn.Module):
    def __init__(self, in_channels, reduction_factor):
        super(SpatialRestorationUnit, self).__init__()
        self.reduction_factor = reduction_factor
        # Calculate the padding needed to restore the input spatial dimensions
        # padding = (input_size - 1) % reduction_factor
        self.trans_conv = nn.ConvTranspose2d(in_channels, in_channels,
                                             kernel_size=(reduction_factor, reduction_factor), stride=reduction_factor)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.trans_conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class SplitUnit(nn.Module):
    def __init__(self, in_channels, hidden_channels, reduction_factor=0):
        super(SplitUnit, self).__init__()
        self.spatial = (reduction_factor != 0)
        self.channel = (in_channels > hidden_channels != -1)
        if self.spatial:
            self.spatial_reduction = SpatialReductionUnit(in_channels, reduction_factor=reduction_factor)
            self.spatial_restoration = SpatialRestorationUnit(in_channels, reduction_factor=reduction_factor)
        if self.channel:
            self.channel_reduction = SingleConv(in_channels, hidden_channels)
            self.channel_restoration = SingleConv(hidden_channels, in_channels)

    def forward(self, x, result_dict):
        # print(x.shape)
        result_dict["before_split_label"] = x
        if self.spatial:
            x = self.spatial_reduction(x)
        # print("Reduced", x.shape)
        if self.channel:
            x = self.channel_reduction(x)
        # print("Reduced channel", x.shape)
        data_size = get_used_memory(x)
        result_dict["transferred_datasize"] = data_size
        # print("Results should be:", 56 * 56 * 4)
        # print("Result without batch size is:", data_size / 8)
        # print("Data size after compression:", get_used_memory(x))
        if self.channel:
            x = self.channel_restoration(x)
        # print("Restored channel", x.shape)
        if self.spatial:
            x = self.spatial_restoration(x)
        # print("Restored spatial", x.shape)
        # exit(0)
        # print("Data size after restoration:", get_used_memory(x))
        result_dict["after_split_output"] = x
        return x



class SkipDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transpose_conv = TransposeConv(in_channels, out_channels)
        self.triple_conv = TripleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, skip_connection):
        # print("Before deconv", x.shape)
        x = self.transpose_conv(x)
        # print("After deconv", x.shape)
        x = torch.cat([x, skip_connection], dim=1)
        # print("Concatenated", x.shape)
        return self.triple_conv(x)



# class FinalDecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.transpose_conv = TransposeConv(in_channels)
#         self.final_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
#                       bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         )
#
#     def forward(self, x, skip_connection):
#         x = self.transpose_conv(x)
#         x = torch.cat([x, skip_connection], dim=1)
#         return self.double_conv(x)