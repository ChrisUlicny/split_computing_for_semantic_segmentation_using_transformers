from torchvision.utils import save_image

import metrics
from utils import utils
from TransformerParts import *
from utils.ModelUtils import GeneralModel, SplitUnit, TransposeConv, SkipDecoderBlock
import os
import time
import TransformerTrainingStrategy


class SwinTransformer(nn.Module, GeneralModel):

    # hidden layers -> C
    # layers -> number of transformer blocks ?
    # downscaling factors -> how much we downscale (divide) an image in each stage
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=32, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True, split_points=None, with_exit,
                 split_channels, split_after_patch_merging, reduction_factors):
        super().__init__()
        if split_points is None:
            self.split_points = [False, False, False, False, False, False, False, False, False]
        else:
            self.split_points = split_points
        # if with_exit:
        #     self.exit_split_points = [True, True]

        self.split_after_patch_merging = split_after_patch_merging
        # reduced_dims = [hidden_dim/2, hidden_dim, hidden_dim*2, hidden_dim*4, hidden_dim*2, hidden_dim, hidden_dim/2, num_classes/4]
        # reduced_dims = [int(i) for i in reduced_dims]

        reduced_dims = split_channels

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
        self.reduction_factors = reduction_factors
        in_channels = [channels, hidden_dim, hidden_dim*2, hidden_dim*4]
        hidden_dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8]

        ### example

        if self.split_points[0] and not self.split_after_patch_merging:
            # no channel reduction
            self.split0 = SplitUnit(in_channels=in_channels[0],  hidden_channels=reduced_dims[0], reduction_factor=reduction_factors[0])

        self.stage1 = StageModule(in_channels=in_channels[0], hidden_dimension=hidden_dims[0], layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                                  with_split=(split_after_patch_merging and self.split_points[0]), reduce_channels_to=reduced_dims[0])


        if self.split_points[1] and not self.split_after_patch_merging:
            # no channel reduction
            self.split1 = SplitUnit(in_channels=in_channels[1],  hidden_channels=reduced_dims[1], reduction_factor=reduction_factors[1])

        self.stage2 = StageModule(in_channels=in_channels[1], hidden_dimension=hidden_dims[1], layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                                  with_split=(split_after_patch_merging and self.split_points[1]), reduce_channels_to=reduced_dims[1])


        if self.split_points[2] and not self.split_after_patch_merging:
            self.split2 = SplitUnit(in_channels=in_channels[2],  hidden_channels=reduced_dims[2], reduction_factor=reduction_factors[2])

        self.stage3 = StageModule(in_channels=in_channels[2], hidden_dimension=hidden_dims[2], layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                                  with_split=(split_after_patch_merging and self.split_points[2]), reduce_channels_to=reduced_dims[2])


        if self.split_points[3] and not self.split_after_patch_merging:
            # no channel reduction
            self.split3 = SplitUnit(in_channels=in_channels[3],  hidden_channels=reduced_dims[3], reduction_factor=reduction_factors[3])

        self.stage4 = StageModule(in_channels=in_channels[3], hidden_dimension=hidden_dims[3], layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                                  with_split=(split_after_patch_merging and self.split_points[3]), reduce_channels_to=reduced_dims[3])


        # self.encoder_modules = nn.ModuleList()

        # for i in range(0, 4):
        #     if split_points[i] and not split_after_patch_merging:
        #         split_unit = SplitUnit(in_channels=in_channels[i], hidden_channels=reduced_dims[i],
        #                                reduction_factor=reduction_factors[i])
        #         self.encoder_modules.append(split_unit)
        #
        #     stage_module = StageModule(in_channels=in_channels[i],
        #                                hidden_dimension=hidden_dims[i],
        #                                layers=layers[i],
        #                                downscaling_factor=downscaling_factors[i],
        #                                num_heads=heads[i],
        #                                head_dim=head_dim,
        #                                window_size=window_size,
        #                                relative_pos_embedding=relative_pos_embedding,
        #                                with_split=(split_after_patch_merging and split_points[i]),
        #                                reduce_channels_to=reduced_dims[i])
        #     self.encoder_modules.append(stage_module)



        if self.split_points[4]:
            self.split4 = SplitUnit(hidden_dim * 8, reduced_dims[4], reduction_factor=reduction_factors[4])
            self.encoder_modules.append(self.split4)


        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(hidden_dim * 8),
        #     nn.Linear(hidden_dim * 8, num_classes)
        # )


        # self.decoder1 = nn.Sequential(TransposeConv(hidden_dim=hidden_dim * 8),
        #                               TripleConv(in_channels=hidden_dim * 8, out_channels=hidden_dim * 4))

        self.decoder1 = SkipDecoderBlock(in_channels=hidden_dim * 8, out_channels=hidden_dim * 4)


        if self.split_points[5]:
            self.split5 = SplitUnit(hidden_dim * 4, reduced_dims[5], reduction_factor=reduction_factors[5])

        # self.decoder2 = nn.Sequential(TransposeConv(hidden_dim=hidden_dim * 4),
        #                               TripleConv(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2))

        self.decoder2 = SkipDecoderBlock(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2)

        if self.split_points[6]:
            self.split6 = SplitUnit(hidden_dim * 2, reduced_dims[6], reduction_factor=reduction_factors[6])

        # self.decoder3 = nn.Sequential(TransposeConv(hidden_dim=hidden_dim * 2),
        #                               TripleConv(in_channels=hidden_dim * 2, out_channels=hidden_dim))
        self.decoder3 = SkipDecoderBlock(in_channels=hidden_dim * 2, out_channels=hidden_dim)


        if self.split_points[7]:
            self.split7 = SplitUnit(hidden_dim, reduced_dims[7], reduction_factor=reduction_factors[7])

        self.decoder4 = nn.Sequential(TransposeConv(in_channels=hidden_dim, out_channels=hidden_dim, stride=4, kernel_size=(4, 4),padding=0),
                                      nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1,
                                                bias=False),
                                      nn.BatchNorm2d(hidden_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(hidden_dim, num_classes, kernel_size=3, padding=1),)

        # if self.split_points[8]:
        #     self.split8 = SplitUnit(num_classes, reduced_dims[8], reduction_factor=8)

    def forward(self, x):

        # RGB -> grayscale
        # transferred_data_sizes = []
        # split_output = None
        exit_split = self.strategy.exit_split()
        skip_connections = []

        self.results = {
            "ee1": None,
            "backbone": None,
            "before_split_label": None,
            "after_split_output": None,
            "transferred_datasize": None
        }



        # for module in self.encoder_modules:
        #     x = module(x, self.results) if isinstance(module, (SplitUnit, StageModule)) else module(x)

        #
        if self.split_points[0] and not self.split_after_patch_merging:
            x = self.split0(x, self.results)
            if exit_split:
                return self.results["before_split_label"]

        # print("Before first stage shape:", img.shape)
        # store split results inside stage
        x = self.stage1(x, self.results, exit_split)
        skip_connections.append(x)
        # if self.results["before_split_label"] is not None and exit_split:
        #     return x

        # print("After first stage shape:", x.shape)

        if self.split_points[1] and not self.split_after_patch_merging:
            x = self.split1(x, self.results)
            if exit_split:
                return self.results["before_split_label"]

        # print(x.shape)
        x = self.stage2(x, self.results, exit_split)
        skip_connections.append(x)
        # if self.results["before_split_label"] is not None and exit_split:
        #     return x
        # print("After second stage shape:", x.shape)

        if self.split_points[2] and not self.split_after_patch_merging:
            x = self.split2(x, self.results)
            if exit_split:
                return self.results["before_split_label"]

        # if self.strategy.enter_final():

        x = self.stage3(x, self.results, exit_split)
        skip_connections.append(x)
        # if self.results["before_split_label"] is not None and exit_split:
        #     return x

        if self.split_points[3] and not self.split_after_patch_merging:
            x = self.split3(x, self.results)
            if exit_split:
                return self.results["before_split_label"]

        x = self.stage4(x, self.results, exit_split)
        # skip_connections.append(x)
        # if self.results["before_split_label"] is not None and exit_split:
        #     return x

        # print("After fourth stage shape:", x.shape)
        if self.split_points[4]:
            x = self.split4(x, self.results)
            if exit_split:
                return self.results["before_split_label"]

            # print(x.shape)
            # x = x.mean(dim=[2, 3])
            # return self.mlp_head(x)

        skip_connections = skip_connections[::-1]
        x = self.decoder1(x, skip_connections[0])

        if self.split_points[5]:
            x = self.split5(x, self.results)
            if exit_split:
                return self.results["before_split_label"]


        # x = self.decoder2(x)
        x = self.decoder2(x, skip_connections[1])

        if self.split_points[6]:
            x = self.split6(x, self.results)
            if exit_split:
                return self.results["before_split_label"]

        # x = self.decoder3(x)
        x = self.decoder3(x, skip_connections[2])

        if self.split_points[7]:
            x = self.split7(x, self.results)
            if exit_split:
                return self.results["before_split_label"]

        x = self.decoder4(x)


        # sending compressed predictions back from cloud to mobile (not used when whole model runs on mobile)
        # if self.split_points[8]:
        #     before_split_label = x
        #     x, compressed_data_size = self.split8(x)
        #     transferred_data_sizes.append(compressed_data_size)
        #     if self.strategy.exit_split:
        #         split_output = [x, before_split_label]

        self.results["backbone"] = x
        # print(self.results)
        # exit('Finished transformer forward')
        result = self.strategy.exit_final(self, self.results)
        return result



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



