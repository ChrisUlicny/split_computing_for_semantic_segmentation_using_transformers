import os.path

from torch.optim.lr_scheduler import PolynomialLR

from models.SegNetWithTwoExitsJoint import *
from models.SwinTransformer import SwinTransformer
from models.SwinLUNET import SwinLUNET
from models.SwinUNET import SwinTransformerSys
from training import *
from models.SegNetWithTwoExits import *
from utils import utils
from loaders import *
from utils.utils import *
from TrainingConfig import ModelTypes, OptimizerType
from models.setr.SETR import SETR_PUP, SETR_PUP_L, SETR_PUP_S
from models.OfficialSwinTransformer import SwinTransformerO


def full_training(config, result_folder, trained_models_folder, splits_to_compute, mp_model_num, min_datasize=None, max_datasize=None):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Set the current working directory to the script directory
    os.chdir(script_dir)
    cwd = os.getcwd()
    # device = f"cuda" if torch.cuda.is_available() else "cpu"
    # device = f"cuda:{str(config.gpu_num)}" if torch.cuda.is_available() else "cpu"
    # warnings.filterwarnings("ignore", category=RuntimeWarning)
    # print('Device:', device)
    # utils.create_device(device)

    print("Saving models to", result_folder)
    device = utils.get_device()
    train_model = config.train_model
    if train_model:
        set_training_stages = config.training_stages
    else:
        set_training_stages = []

    make_prediction = config.make_prediction
    epochs = config.num_epochs

    num_workers = 0
    pin_memory = True


    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    optimizer_type = config.optimizer_type
    if optimizer_type == OptimizerType.SGD:
        momentum = config.momentum
        weight_decay = config.weight_decay
    else:
        momentum = None

    model_num = 1

    # must be dividable by window size (7 in our case)
    if config.training_type == ModelTypes.SETR:
        train_transform, val_transform = get_augmentations_new(img_height=768, img_width=768)
    else:
        train_transform, val_transform = get_augmentations_new(img_height=224, img_width=224)

    # train_loader, val_loader, test_loader, color_map = get_loaders_camvid(
    #     batch_size,
    #     train_transform,
    #     val_transform,
    #     num_workers,
    #     pin_memory
    # )

    train_loader, val_loader, test_loader, color_map = get_loaders_cityscapes(
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory
    )


    resume_to_split = config.split_index
    for split_index in range(resume_to_split, len(splits_to_compute)):
        config.split_index = split_index
        print("training:", splits_to_compute[split_index])
        if mp_model_num is not None:
            model_num = mp_model_num
        else:
            model_num = 1
        model_name = config.model_name if "{}" not in config.model_name else config.model_name.format(config.channels[split_index], split_index)
        if not os.path.exists(os.path.join(result_folder, model_name)):
            os.makedirs(os.path.join(result_folder, model_name))
        if hasattr(config, 'splits'):
            config.splits = str(splits_to_compute[split_index])
        training_type = config.training_type
        load_model = config.load_model
        same_models_num = config.repeat_model
        for same_model in range(1, same_models_num + 1):
            print(f"Training model {same_model} out of {same_models_num}")

            total_time = 0
            train_acc_list = []
            val_acc_list = []
            train_miou_list = []
            val_miou_list = []
            train_loss_list = []
            val_loss_list = []
            learning_rates = []
            if training_type == ModelTypes.SEPARATE:
                model = SegNetWithTwoExits(in_channels=3, out_channels=32)
            elif training_type == ModelTypes.JOINT:
                model = SegNetWithTwoExitsJoint(in_channels=3, out_channels=32)
            elif training_type == ModelTypes.TRANSFORMER:
                model = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), channels=3,
                                        num_classes=20, head_dim=32, window_size=7, downscaling_factors=(4, 2, 2, 2),
                                        relative_pos_embedding=True, split_points=splits_to_compute[split_index],
                                        with_exit=False, split_channels=config.channels, split_after_patch_merging=False,
                                        reduction_factors=config.reduction_factors)
            elif training_type == ModelTypes.SWINLUNET:
                model = SwinLUNET(num_classes=20, split_points=splits_to_compute[split_index],
                                        with_exit=False, split_channels=config.channels)
            elif training_type == ModelTypes.SWINUNET:
                model = SwinTransformerSys(img_size=224, patch_size=4, in_chans=3, num_classes=20)
            elif training_type == ModelTypes.SETR:
                # model = SETR_PUP(img_dim=768, patch_dim=16, num_channels=3, num_classes=20, embedding_dim=1024, num_heads=16, num_layers=24,
                #                  hidden_dim=4 * 1024)
                _, model = SETR_PUP_L()
            elif training_type == ModelTypes.SWINOFFICIAL:
                model = SwinTransformerO()
            else:
                raise Exception("Unexpected training type")
            model = model.to(device=device)
            for stage in set_training_stages:
                model.set_strategy(stage)
                # for early stopping
                max_eval = 0

                loss_func = model.get_loss_func()
                optimizer = model.initialize_optimizer(learning_rate, optimizer_type, momentum=momentum)
                scheduler = PolynomialLR(optimizer, total_iters=epochs, power=0.9)
                prev_epoch = 0
                costs = []
                costs = model.save_costs(costs)
                if load_model:
                    model_state_dict = model.state_dict()
                    print("\nModel state dict keys and shapes:")
                    for key, value in model_state_dict.items():
                        print(f"{key}: {value.shape}")
                    print("Loading model")
                    print("\n\n\n\n")
                    loaded_checkpoint = model.load_checkpoint(path=f"{trained_models_folder}/{config.model_to_load}.pth.tar")
                    _= model.load_pretrained_checkpoint(path=f"{trained_models_folder}/{config.model_to_load}")
                    if config.resuming_training:

                            train_acc_list = loaded_checkpoint["all_train_accs"]
                            val_acc_list = loaded_checkpoint["all_val_accs"]
                            train_miou_list = loaded_checkpoint["all_train_mious"]
                            val_miou_list = loaded_checkpoint["all_val_mious"]
                            train_loss_list = loaded_checkpoint["all_train_losses"]
                            val_loss_list = loaded_checkpoint["all_val_losses"]
                            learning_rates = loaded_checkpoint["learning_rates"]
                            total_time = loaded_checkpoint["total_time"]
                            optimizer.load_state_dict(loaded_checkpoint["optimizer"])
                            scheduler.load_state_dict(loaded_checkpoint["scheduler"])
                            prev_epoch = loaded_checkpoint["epoch"]

                # else:

                # evaluation before training
                # print('Evaluation before training..')
                # val_accuracy, val_miou, val_loss = model.check_batch_acc_miou_loss(val_loader, loss_func=loss_func)
                # train_accuracy, train_miou, train_loss = model.check_batch_acc_miou_loss(train_loader, loss_func=loss_func)
                # metric_dict = {
                #     "val_acc_list": val_acc_list,
                #     "train_acc_list": train_acc_list,
                #     "val_miou_list": val_miou_list,
                #     "train_miou_list": train_miou_list,
                #     "val_loss_list": val_loss_list,
                #     "train_loss_list": train_loss_list,
                #     "val_acc": val_accuracy,
                #     "train_acc": train_accuracy,
                #     "val_miou": val_miou,
                #     "train_miou": train_miou,
                #     "val_loss": val_loss,
                #     "train_loss": train_loss
                # }
                # metric_dict = model.append_metric_lists(metric_dict, 0)
                # train_accuracy = metric_dict.get("train_acc")
                # train_miou = metric_dict.get("train_miou")
                # train_loss = metric_dict.get("train_loss")
                # val_loss = metric_dict.get("val_loss")
                # val_accuracy = metric_dict.get("val_acc")
                # val_miou = metric_dict.get("val_miou")
                # train_acc_list = metric_dict.get("train_acc_list")
                # val_acc_list = metric_dict.get("val_acc_list")
                # train_miou_list = metric_dict.get("train_miou_list")
                # val_miou_list = metric_dict.get("val_miou_list")
                # train_loss_list = metric_dict.get("train_loss_list")
                # val_loss_list = metric_dict.get("val_loss_list")
                #
                # checkpoint = {
                #     "epoch": 0,
                #     "train_loss": train_loss,
                #     "train_accuracy": train_accuracy,
                #     "train_miou": train_miou,
                #     "val_loss": val_loss,
                #     "val_accuracy": val_accuracy,
                #     "val_miou": val_miou,
                #     "all_train_accs": train_acc_list,
                #     "all_val_accs": val_acc_list,
                #     "all_train_mious": train_miou_list,
                #     "all_val_mious": val_miou_list,
                #     "all_train_losses": train_loss_list,
                #     "all_val_losses": val_loss_list,
                #     "learning_rates": [learning_rate]
                # }
                # model.save_checkpoint(checkpoint, filename=model_name, folder=result_folder, save_model=False,
                #                       model_num=model_num)

                if train_model:
                    config.save_to_json(f"{result_folder}/{model_name}/training_config.json")
                    for epoch in range(1, epochs+1):
                        if epoch > config.num_epochs:
                            break

                        print('Epoch:', prev_epoch+epoch, '/', prev_epoch + epochs)
                        train_accuracy, train_miou, train_loss = train(train_loader, model, optimizer, loss_func)
                        costs = model.save_costs(costs)
                        learning_rates.append(optimizer.param_groups[0]["lr"])
                        print('Calculating validation results')
                        val_accuracy, val_miou, val_loss = model.check_batch_acc_miou_loss(val_loader, loss_func=loss_func)
                        # print('Calculating training results')
                        # train_accuracy, train_miou, train_loss = model.check_batch_acc_miou_loss(train_loader, loss_func=loss_func)
                        # train_accuracy, train_miou, train_loss = [0, 0, 0], [0, 0, 0], [0, 0, 0]

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

                        metric_dict = model.append_metric_lists(metric_dict, epoch)


                        scheduler.step()
                        # optimizer.param_groups[-1]["lr"] = 0.001
                        # scheduler.step(val_miou[0])

                        # train_loss, train_accuracy, train_miou = [0, 0, 0], [0, 0, 0], [0, 0, 0]
                        # val_loss, val_accuracy, val_miou = [0, 0, 0], [0, 0, 0], [0, 0, 0]
                        train_accuracy = metric_dict.get("train_acc")
                        train_miou = metric_dict.get("train_miou")
                        train_loss = metric_dict.get("train_loss")
                        val_loss = metric_dict.get("val_loss")
                        val_accuracy = metric_dict.get("val_acc")
                        val_miou = metric_dict.get("val_miou")
                        train_acc_list = metric_dict.get("train_acc_list")
                        val_acc_list = metric_dict.get("val_acc_list")
                        train_miou_list = metric_dict.get("train_miou_list")
                        val_miou_list = metric_dict.get("val_miou_list")
                        train_loss_list = metric_dict.get("train_loss_list")
                        val_loss_list = metric_dict.get("val_loss_list")


                        checkpoint = {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": prev_epoch + epoch,
                            "train_loss": train_loss,
                            "train_accuracy": train_accuracy,
                            "train_miou": train_miou,
                            "val_loss": val_loss,
                            "val_accuracy": val_accuracy,
                            "val_miou": val_miou,
                            "all_train_accs": train_acc_list,
                            "all_val_accs": val_acc_list,
                            "all_train_mious": train_miou_list,
                            "all_val_mious": val_miou_list,
                            "all_train_losses": train_loss_list,
                            "all_val_losses": val_loss_list,
                            "total_time": total_time,
                            "learning_rates": learning_rates,
                            "costs": costs
                        }
                        # early stopping mechanism to prevent overtraining
                        if sum(val_miou) > max_eval:
                            model.save_checkpoint(checkpoint, filename=model_name, folder=result_folder, save_model=True, model_num=model_num)
                            max_eval = sum(val_miou)
                        else:
                            model.save_checkpoint(checkpoint, filename=model_name, folder=result_folder, save_model=False,
                                                      model_num=model_num)

                    model.draw_losses_graph_joint(train_acc_list, val_acc_list, f'{result_folder}/{model_name}/acc_graph{model_num}', "Accuracy")
                    model.draw_losses_graph_joint(train_miou_list, val_miou_list, f'{result_folder}/{model_name}/mious_graph{model_num}', "mIoU")
                    model.draw_losses_graph_joint(train_loss_list, val_loss_list,
                                                  f'{result_folder}/{model_name}/losses_graph{model_num}', "Loss")
                    utils.draw_learning_rate(learning_rates, f'{result_folder}/{model_name}/lr_graph{model_num}')






                # passing one image input
            if make_prediction:
                # print("Loading model")
                # loaded_checkpoint = model.load_checkpoint(
                #     path=f"{trained_models_folder}/{config.model_to_load}.pth.tar")
                model.set_strategy(TransformerTrainingStrategy.TransformerInferenceStrategy())
                full_prediction_folder = f"{result_folder}/{model_name}/predictions{model_num}"
                if not os.path.exists(full_prediction_folder):
                    os.makedirs(full_prediction_folder)
                times = []


                cumulate_t_uplink = 0
                cumulate_t_downlink = 0

                for idx, (image, mask) in enumerate(test_loader):
                    image = image.to(device=device)
                    mask = mask.to(device=device)
                    # if idx == 0:
                    #     flops = FlopCountAnalysis(model, image)
                    #     print("total Flops:", flops.total())
                    #     print("Flops:", flop_count_table(flops))
                    #     # print("Accessing structure:", flops.by_module())
                    #
                    #     mobile_flops, server_flops = calculate_before_and_after_flops(flops.by_module(), splits_to_compute[split_index])
                    #     array_of_mobile_flops.append(mobile_flops)
                    #     array_of_cloud_flops.append(server_flops)
                    #
                    #     print("total Flops:", flops.total())
                    #     print("mobile Flops:", mobile_flops)
                    #     print("server Flops:", server_flops)
                    #     exit(0)


                    # start_time = time.time()
                    pred_for_acc, time_elapsed = model.make_prediction(image, mask, model, idx + 1, color_map, full_prediction_folder)
                    if idx == 5:
                        break
                    # end_time = time.time()

                    for i in pred_for_acc:
                        acc = utils.check_acc_one_pic(i, mask)
                        # array_of_accs.append(acc)
                        mask_squeezed = torch.squeeze(mask)
                        one_pic_miou = utils.check_miou_one_pic(i, mask_squeezed, num_classes=model.num_classes)
                        # one_pic_miou2 = utils.check_miou_one_pic_new(i, mask_squeezed, num_classes=model.num_classes)
                        # one_pic_miou3 = utils.check_miou_one_pic_third(i, mask_squeezed, num_classes=model.num_classes)
                        # one_pic_miou3 = utils.check_miou_one_pic_four(i, mask_squeezed, num_classes=model.num_classes)
                        # array_of_mious.append(one_pic_miou)

                    # print("The prediction took", round(time_elapsed, 2), "seconds")
                    # times.append(round(time_elapsed, 2))



            model_num += 1

