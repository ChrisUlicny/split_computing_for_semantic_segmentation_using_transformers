from torchvision.utils import save_image

import metrics
from utils import utils
from TransformerParts import *
from utils.ModelUtils import GeneralModel2
import os
import time
import TransformerTrainingStrategy


# maybe add argument to determine if compression or decompression should take place

class SwinLUNET(GeneralModel2):

    # hidden layers -> C
    # layers -> number of transformer blocks ?
    # downscaling factors -> how much we downscale (divide) an image in each stage
    def __init__(self, *, hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), channels=3, num_classes=20, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True, split_points= None, with_exit=False, split_channels=None):
        super().__init__()

        self.strategy = None

        self.num_classes = num_classes


        self.results = {
            "ee1": None,
            "backbone": None,
            "before_split_label": None,
            "after_split_output": None,
            "transferred_datasize": [0]
        }

        self.datasizes = []
        in_channels = [channels, hidden_dim, hidden_dim*2, hidden_dim*4]
        hidden_dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8]

        self.stage1 = StageModule(in_channels=in_channels[0], hidden_dimension=hidden_dims[0], layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                                  with_split=False, embed=True)


        self.stage2 = StageModule(in_channels=in_channels[1], hidden_dimension=hidden_dims[1], layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                                  with_split=False)


        self.stage3 = StageModule(in_channels=in_channels[2], hidden_dimension=hidden_dims[2], layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                                  with_split=False)



        self.stage4 = StageModule(in_channels=in_channels[3], hidden_dimension=hidden_dims[3], layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                                  with_split=False)



        # self.decoder1 = SkipDecoderBlock(in_channels=hidden_dim * 8, out_channels=hidden_dim * 4)
        self.decoder1 = SkipDecoderStageModule(in_channels=hidden_dim * 8, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.decoder2 = SkipDecoderStageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.decoder3 = SkipDecoderStageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim, layers=layers[0],
                                  num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.final_patch_expansion = FinalPatchExpansion(out_channels=hidden_dims[0])

        self.output = nn.Conv2d(in_channels=hidden_dims[0], out_channels=self.num_classes, kernel_size=1, bias=False)


    def forward(self, x):

        exit_split = self.strategy.exit_split()

        skip_connections = []

        self.results = {
            "ee1": None,
            "backbone": None,
            "before_split_label": None,
            "after_split_output": None,
            "transferred_datasize": None
        }


        x = self.stage1(x, self.results, exit_split)
        skip_connections.append(x)


        x = self.stage2(x, self.results, exit_split)
        skip_connections.append(x)


        x = self.stage3(x, self.results, exit_split)
        skip_connections.append(x)

        x = self.stage4(x, self.results, exit_split)

        # x = self.norm(x)


        skip_connections = skip_connections[::-1]

        x = self.decoder1(x, skip_connections[0])

        # x = self.decoder2(x)
        x = self.decoder2(x, skip_connections[1])

        # x = self.decoder3(x)
        x = self.decoder3(x, skip_connections[2])

        # x = self.norm_up(x)

        x = self.final_patch_expansion(x)
        x = self.output(x.permute(0,3,1,2))

        self.results["backbone"] = x
        return self.results["backbone"]


    def initialize_optimizer(self, learning_rate, optimizer_type, momentum=0.95):
        return self.strategy.initialize_optimizer(self, learning_rate, optimizer_type, momentum)

    def save_checkpoint(self, state, filename, folder, save_model, model_num=0):
        utils.save_checkpoint_split_json(state, filepath=f"{folder}/{filename}", save_model=save_model, model_num=model_num, exits_num=1)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.load_state_dict(state_dict=checkpoint["state_dict"], strict=False)
        return checkpoint


    def check_batch_acc_miou_loss(self, loader, loss_func):
        # if self.training_strategy != TrainingState.JOINT and self.training_strategy != TrainingState.DISTILLATION_JOINT\
        #         and self.training_strategy != TrainingState.EXIT_ENSEMBLE:
        return self.strategy.check_batch_acc_miou_loss(self, loader, self.num_classes, loss_func)
        # else:

    def check_batch_acc_miou_loss_no_loader(self, x, y):
        return self.strategy.check_batch_acc_miou_no_loader(self, x, y, self.num_classes)

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

    def reset_exit_counts(self):
        pass

    def draw_losses_graph_joint(self, train_acc_list, val_acc_list, file, metric_type, exits_num=None):
        utils.draw_losses_graph_backbone(train_acc_list, val_acc_list, file, metric_type)

    def get_split_point(self):
        return self.split_points


    def make_prediction(self, image, mask, model, idx, color_map, prediction_folder):
        if not os.path.exists(prediction_folder):
            # if the directory is not present, create it
            os.makedirs(prediction_folder)
        save_image(image, f"{prediction_folder}/original{idx}.png")
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
        # print('prediction size pre argmax after model:', prediction.shape)
        prediction = torch.argmax(prediction, dim=0)
        # print('prediction size:', prediction.shape)
        new_image = utils.seg_map_to_image(prediction, color_map)
        # print('new image size:', new_image.shape)
        utils.save_numpy_as_image(new_image, f"{prediction_folder}/pred{idx}.png")
        time_elapsed = end - start
        return prediction_for_accuracy, time_elapsed

    def get_datasizes(self, mb):
        if mb is True:
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



