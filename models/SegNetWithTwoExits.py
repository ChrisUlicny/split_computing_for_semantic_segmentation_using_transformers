from utils.ModelUtils import *
import metrics
from misc import ForwardStrategy
import os
from utils import utils
import PoolingStrategy

class SegNetWithTwoExits(nn.Module, GeneralModel):

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 512]):
        super(SegNetWithTwoExits, self).__init__()
        # storing conv layers
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.first_exit_ups = nn.ModuleList()
        self.exit_ups = nn.ModuleList()
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.pooling_strategy = PoolingStrategy.MaxPoolingStrategy()
        self.exits = [None, None, None]
        self.exit_counts = [0, 0, 0]
        self.dynamic_weights = False
        self.thresholds = [0.83, 0.84]
        self.num_classes = out_channels
        self.training_strategy = None
        self.pooling_convs = []
        self.unpooling_convs = []


        # DOWN part of SegNet
        self.downs.append(DoubleConv(in_channels=in_channels, out_channels=features[0]))
        self.downs.append(DoubleConv(in_channels=features[0], out_channels=features[1]))
        self.downs.append(TripleConv(in_channels=features[1], out_channels=features[2]))

        # first exit UP part
        self.first_exit_ups.append(DoubleConv(in_channels=features[2], out_channels=features[1]))
        self.first_exit_ups.append(DoubleConv(in_channels=features[1], out_channels=features[0]))
        self.first_exit_ups.append(nn.Sequential(
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        ))

        # rest of down part
        self.downs.append(TripleConv(in_channels=features[2], out_channels=features[3]))
        self.downs.append(TripleConv(in_channels=features[3], out_channels=features[4]))

        # UP part of second exit - AlexNet after third conv
        self.exit_ups.append(SingleConv(in_channels=512, out_channels=512))
        self.exit_ups.append(SingleConv(in_channels=512, out_channels=256))
        self.exit_ups.append(SingleConv(in_channels=256, out_channels=128))
        self.exit_ups.append(SingleConv(in_channels=128, out_channels=64, kernel_size=5, padding=2))
        self.final_exit_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # UP part of SegNet
        self.ups.append(TripleConv(in_channels=features[4], out_channels=features[3]))
        self.ups.append(TripleConv(in_channels=features[3], out_channels=features[2]))
        self.ups.append(TripleConv(in_channels=features[2], out_channels=features[1]))
        self.ups.append(DoubleConv(in_channels=features[1], out_channels=features[0]))
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )
        # self.ups.append(self.final_conv)

        if isinstance(self.pooling_strategy, PoolingStrategy.ConvPoolingStrategy):
            self.pooling_convs.append(nn.Conv2d(features[0], features[0], kernel_size=2, stride=2))
            self.pooling_convs.append(nn.Conv2d(features[1], features[1], kernel_size=2, stride=2))
            self.pooling_convs.append(nn.Conv2d(features[2], features[2], kernel_size=2, stride=2))
            self.pooling_convs.append(nn.Conv2d(features[3], features[3], kernel_size=2, stride=2))

            self.unpooling_convs.append(TransposeConv(features[3]))
            self.unpooling_convs.append(TransposeConv(features[2]))
            self.unpooling_convs.append(TransposeConv(features[1]))
            self.unpooling_convs.append(TransposeConv(features[0]))
        else:
            self.pooling_convs = 5 * [None]
            self.unpooling_convs = 5 * [None]

    # removed last down sampling and first up sampling -> was useless
    def forward(self, x):
        # training
        # encoding
        if self.dynamic_weights:
            self.weights = torch.sigmoid(self.costs)
        enter_first_exit = self.training_strategy.enter_first()
        enter_second_exit = self.training_strategy.enter_second()
        enter_final_exit = self.training_strategy.enter_final()
        indices, sizes = [], []
        for idx, down in enumerate(self.downs):
            x = down(x)

            # first exit
            if enter_first_exit and idx == 2:
                # flipping for decoding
                indices = indices[::-1]
                sizes = sizes[::-1]
                x_first_exit = self.first_exit_ups[0](x)
                output_size = sizes[0]
                # x_first_exit = self.unpool(input=x_first_exit, indices=indices[0], output_size=output_size)
                x_first_exit = self.pooling_strategy.do_unpooling(inputs=x_first_exit, indices=indices[0], output_size=output_size, layer=self.unpooling_convs[2])
                x_first_exit = self.first_exit_ups[1](x_first_exit)
                output_size = sizes[1]
                # x_first_exit = self.unpool(input=x_first_exit, indices=indices[1], output_size=output_size)
                x_first_exit = self.pooling_strategy.do_unpooling(inputs=x_first_exit, indices=indices[1],
                                                                  output_size=output_size, layer=self.unpooling_convs[3])
                x_first_exit = self.first_exit_ups[2](x_first_exit)
                self.exits[ForwardStrategy.ExitPosition.FIRST_EXIT.value] = x_first_exit
                # flipping back for the rest of the down section
                indices = indices[::-1]
                sizes = sizes[::-1]
                result = self.training_strategy.exit_first(self, self.exits)
                if result is not None:
                    return result

            # disabling last pooling layer
            if idx < 4:
                sizes.append(x.size())
                # x, ind = self.pool(x)
                x, ind = self.pooling_strategy.do_pooling(x, layer=self.pooling_convs[idx])
                indices.append(ind)

        # flipping for decoding
        indices = indices[::-1]
        sizes = sizes[::-1]
        # cutting last added size which is not used anyway

        # early exit -> AlexNet-like
        if enter_second_exit:
            # reducing channels 512 -> 256
            x_second_exit = self.exit_ups[0](x)
            # first unpool
            output_size = sizes[0]
            # x_second_exit = self.unpool(input=x_second_exit, indices=indices[0], output_size=output_size)
            x_second_exit = self.pooling_strategy.do_unpooling(inputs=x_second_exit, indices=indices[0], output_size=output_size, layer=self.unpooling_convs[0])
            x_second_exit = self.exit_ups[1](x_second_exit)
            output_size = sizes[1]
            # x_second_exit = self.unpool(input=x_second_exit, indices=indices[1], output_size=output_size)
            x_second_exit = self.pooling_strategy.do_unpooling(inputs=x_second_exit, indices=indices[1],
                                                               output_size=output_size, layer=self.unpooling_convs[1])
            x_second_exit = self.exit_ups[2](x_second_exit)
            output_size = sizes[2]
            # x_second_exit = self.unpool(input=x_second_exit, indices=indices[2], output_size=output_size)
            x_second_exit = self.pooling_strategy.do_unpooling(inputs=x_second_exit, indices=indices[2],
                                                               output_size=output_size, layer=self.unpooling_convs[2])
            x_second_exit = self.exit_ups[3](x_second_exit)
            output_size = sizes[3]
            # x_second_exit = self.unpool(input=x_second_exit, indices=indices[3], output_size=output_size)
            x_second_exit = self.pooling_strategy.do_unpooling(inputs=x_second_exit, indices=indices[3],
                                                               output_size=output_size, layer=self.unpooling_convs[3])
            x_second_exit = self.final_exit_conv(x_second_exit)
            self.exits[ForwardStrategy.ExitPosition.SECOND_EXIT.value] = x_second_exit
            result = self.training_strategy.exit_second(self, self.exits)
            if result is not None:
                return result

        if enter_final_exit:
            # decoding part of segnet
            x = self.ups[0](x)
            # 2nd block
            output_size = sizes[0]
            # x = self.unpool(input=x, indices=indices[0], output_size=output_size)
            x = self.pooling_strategy.do_unpooling(inputs=x, indices=indices[0],
                                                               output_size=output_size, layer=self.unpooling_convs[0])
            x = self.ups[1](x)
            # 3rd block
            output_size = sizes[1]
            # x = self.unpool(input=x, indices=indices[1], output_size=output_size)
            x = self.pooling_strategy.do_unpooling(inputs=x, indices=indices[1],
                                                   output_size=output_size, layer=self.unpooling_convs[1])
            x = self.ups[2](x)
            # 4th block
            output_size = sizes[2]
            # x = self.unpool(input=x, indices=indices[2], output_size=output_size)
            x = self.pooling_strategy.do_unpooling(inputs=x, indices=indices[2],
                                                   output_size=output_size, layer=self.unpooling_convs[2])
            x = self.ups[3](x)
            # 5th block
            output_size = sizes[3]
            # x = self.unpool(input=x, indices=indices[3], output_size=output_size)
            x = self.pooling_strategy.do_unpooling(inputs=x, indices=indices[3],
                                                   output_size=output_size, layer=self.unpooling_convs[3])
            x = self.final_conv(x)
            self.exits[ForwardStrategy.ExitPosition.BACKBONE.value] = x
        result = self.training_strategy.exit_final(self, self.exits)
        if result is not None:
            return result

    def print_exits_taken(self):
        print(f'Took first exit: {self.exit_counts[0]}\nTook second exit: {self.exit_counts[1]}\nTook final exit: {self.exit_counts[2]}')

    def initialize_optimizer(self, learning_rate, optimizer_type, momentum=0.95):
        return self.training_strategy.initialize_optimizer(self, optimizer_type, learning_rate, momentum=momentum)

    def save_checkpoint(self, state, filename, folder, save_model, model_num=0):
        utils.save_checkpoint_joint(state, filename, folder, save_model, model_num)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.load_state_dict(state_dict=checkpoint["state_dict"], strict=False)
        return checkpoint

    def load_pretrained_cityscapes(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        pretrained_model = checkpoint["state_dict"]
        pretrained_model.pop('final_conv.3.weight', None)
        pretrained_model.pop('final_conv.3.bias', None)
        self.load_state_dict(state_dict=checkpoint["state_dict"], strict=False)
        optim_state = checkpoint["optimizer"]
        epoch = 0
        return epoch, optim_state

    def check_batch_acc_miou_loss(self, loader, loss_func):
        # if self.training_strategy != TrainingState.JOINT and self.training_strategy != TrainingState.DISTILLATION_JOINT\
        #         and self.training_strategy != TrainingState.EXIT_ENSEMBLE:
        return metrics.check_batch_acc_miou_loss(self, loader, self.num_classes, loss_func)
        # else:

    def check_batch_miou(self, loader,  num_classes=None):
        # if self.training_strategy != TrainingState.JOINT and self.training_strategy != TrainingState.DISTILLATION_JOINT\
        #         and self.training_strategy != TrainingState.EXIT_ENSEMBLE:
        return metrics.check_batch_miou(self, loader, self.num_classes)
        # else:

    def get_loss_func(self):
        return self.training_strategy.get_loss_func()

    def set_strategy(self, strategy: ForwardStrategy):
        if isinstance(strategy, ForwardStrategy.ExitEnsemble) or isinstance(strategy, ForwardStrategy.OnlineDistillation) \
                or isinstance(strategy, ForwardStrategy.JointTrainingStrategy):
            raise Exception("Invalid training strategy for frozen backbone training")
        else:
            self.training_strategy = strategy

    def save_hyperparameters(self, hyperparameters: dict, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = f"{path}/hyperparameters.txt"
        with open(full_path, 'w') as file:
            self.training_strategy.write_hyperparameters_to_file(hyperparameters, file)
            file.write('\n')

    def reset_exit_counts(self):
        self.exit_counts = [0, 0, 0]

    def calculate_loss(self, predictions, labels, loss_func):
        return self.training_strategy.calculate_loss(predictions, labels, loss_func)

    def make_prediction(self, image, mask, model, idx, color_map, prediction_folder):
        return self.training_strategy.make_prediction(image, mask, model, idx, color_map, prediction_folder)

    def draw_losses_graph_joint(self, train_acc_list, val_acc_list, file, metric_type, exits_num=3):
        utils.draw_losses_graph_joint(train_acc_list, val_acc_list, file, metric_type, exits_num=3)

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

        # when training early exits
        if len(val_accuracy) == 2:
            val_loss = [0, val_loss[1], val_loss[0]]
            val_accuracy = [0, val_accuracy[1], val_accuracy[0]]
            val_miou = [0, val_miou[1], val_miou[0]]
            train_loss = [0, train_loss[1], train_loss[0]]
            train_accuracy = [0, train_accuracy[1], train_accuracy[0]]
            train_miou = [0, train_miou[1], train_miou[0]]
            if len(val_acc_list) > epoch:
                val_acc_list[epoch][2] = val_accuracy[2]
                val_acc_list[epoch][1] = val_accuracy[1]
                val_miou_list[epoch][2] = val_miou[2]
                val_miou_list[epoch][1] = val_miou[1]
                val_loss_list[epoch][2] = val_loss[2]
                val_loss_list[epoch][1] = val_loss[1]
                train_acc_list[epoch][2] = train_accuracy[2]
                train_acc_list[epoch][1] = train_accuracy[1]
                train_miou_list[epoch][2] = train_miou[2]
                train_miou_list[epoch][1] = train_miou[1]
                train_loss_list[epoch][2] = train_loss[2]
                train_loss_list[epoch][1] = train_loss[1]
            else:
                val_acc_list.append(val_accuracy)
                val_miou_list.append(val_miou)
                val_loss_list.append(val_loss)
                train_acc_list.append(train_accuracy)
                train_miou_list.append(train_miou)
                train_loss_list.append(train_loss)

        # when training backbone
        elif len(val_accuracy) == 1:
            val_loss = [val_loss[0], 0, 0]
            val_accuracy = [val_accuracy[0], 0, 0]
            val_miou = [val_miou[0], 0, 0]
            train_loss = [train_loss[0], 0, 0]
            train_accuracy = [train_accuracy[0], 0, 0]
            train_miou = [train_miou[0], 0, 0]
            if len(val_acc_list) > epoch:
                val_acc_list[epoch][0] = val_accuracy[0]
                val_miou_list[epoch][0] = val_miou[0]
                train_acc_list[epoch][0] = train_accuracy[0]
                train_miou_list[epoch][0] = train_miou[0]
                train_loss_list[epoch][0] = train_loss[0]
                val_loss_list[epoch][0] = val_loss[0]
            else:
                val_miou_list.append(val_miou)
                val_acc_list.append(val_accuracy)
                train_miou_list.append(train_miou)
                train_acc_list.append(train_accuracy)
                val_loss_list.append(val_loss)
                train_loss_list.append(train_loss)

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
