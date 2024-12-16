import metrics
from utils import utils
from TransformerParts import *
from utils.ModelUtils import GeneralModel, SplitUnit
import os
import TransformerTrainingStrategy


# maybe add argument to determine if compression or decompression should take place

class Autoencoder(nn.Module, GeneralModel):

    def __init__(self, in_channels, reduce_to_channels, reduction_factor, index):
        super().__init__()


        self.strategy = None

        self.results = {
            "before_split_label": None,
            "after_split_output": None,
            "transferred_datasize": [0]
        }

        # self.split = SplitUnit(in_channels=in_channels,  hidden_channels=reduce_to_channels, reduction_factor=reduction_factor)
        # Dynamically name the SplitUnit based on the provided index
        self.split_name = f"split{index}"
        setattr(self, self.split_name, SplitUnit(in_channels, reduce_to_channels, reduction_factor))

    def forward(self, x):

        # exit_split = self.strategy.exit_split()
        # self.results = {
        #     "before_split_label": None,
        #     "after_split_output": None,
        #     "transferred_datasize": [0]
        # }

        split = getattr(self, self.split_name)
        x = split(x, self.results)
        # self.results["before_split_label"] = x
        # self.results["after_split_output"] = x

        return x



    def initialize_optimizer(self, learning_rate, optimizer_type, momentum=0.95):
        return self.strategy.initialize_optimizer(self, learning_rate, optimizer_type, momentum)

    def save_checkpoint(self, state, filename, folder, save_model, model_num=0):
        utils.save_checkpoint_only_split_json(state, filepath=f"{folder}/{filename}", save_model=save_model, model_num=model_num, exits_num=1)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.load_state_dict(state_dict=checkpoint["state_dict"], strict=False)
        return checkpoint


    def check_batch_acc_miou_loss(self, loader, loss_func):
        # if self.training_strategy != TrainingState.JOINT and self.training_strategy != TrainingState.DISTILLATION_JOINT\
        #         and self.training_strategy != TrainingState.EXIT_ENSEMBLE:
        return self.strategy.check_batch_loss(self, loader, loss_func)
        # else:

    def check_batch_acc_miou_loss_no_loader(self, x, y):
        return torch.tensor(0), torch.tensor(0)

    def check_batch_miou(self, loader,  num_classes=None):
        # if self.training_strategy != TrainingState.JOINT and self.training_strategy != TrainingState.DISTILLATION_JOINT\
        #         and self.training_strategy != TrainingState.EXIT_ENSEMBLE:
        return metrics.check_batch_miou(self, loader, self.num_classes)
        # else:

    def get_loss_func(self):
        return self.strategy.get_loss_func()

    def calculate_loss(self, predictions, labels, loss_func):
        return self.strategy.calculate_loss(predictions, labels, loss_func)

    def set_strategy(self, strategy: TransformerTrainingStrategy.TransformerStrategy):
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
            file.write(f"Hyperparameters: start lr={lr}, batch size={str(bs)}, optimizer={optimizer_type}")
            if momentum is not None:
                file.write(f', momentum: {momentum}, weight_decay: {weight_decay}')
            file.write('\n')


    def draw_losses_graph_joint(self, train_acc_list, val_acc_list, file, metric_type, exits_num=None):
        utils.draw_losses_graph_backbone(train_acc_list, val_acc_list, file, metric_type)


    def get_datasizes(self, mb):
        if mb is True:
            self.datasizes = [i / 1048576 for i in self.results["transferred_datasize"]]
            return self.datasizes
        else:
            return self.datasizes


    def append_metric_lists(self, metric_dict, epoch):
        val_loss = metric_dict.get("val_loss")
        train_loss = metric_dict.get("train_loss")
        train_loss_list = metric_dict.get("train_loss_list")
        val_loss_list = metric_dict.get("val_loss_list")


        if len(val_loss_list) > epoch:
            val_loss_list[epoch][0] = val_loss[0]
            train_loss_list[epoch][0] = train_loss[0]
        else:
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)

        metric_dict = {
            "val_loss_list": val_loss_list,
            "train_loss_list": train_loss_list,
            "val_loss": val_loss,
            "train_loss": train_loss
        }
        return metric_dict
