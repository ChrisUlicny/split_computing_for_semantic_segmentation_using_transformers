import concurrent.futures
import gc
import multiprocessing
import os.path

import pandas as pd
from torch.optim.lr_scheduler import PolynomialLR

from models.SwinTransformer import SwinTransformer
from training import *
import time
import utils
from loaders import *
from utils import *
from TrainingConfig import OptimizerType
from TransformerTrainingStrategy import CreateSplitDataset, TrainSplitOnly
from models.Autoencoder import Autoencoder



def print_model_layers(model):
    for name, module in model.named_children():
        print(f"{name} | {module}")
    # for name in model.modules():
    #     print(f"Name: {name}")


def plot_bar_chart(filepath, data_x, data_y, xlabel='Categories', ylabel='Values'):
    categories = data_x
    values = data_y

    title = f'{ylabel} w.r.t {xlabel}'
    plt.bar(categories, values, color='skyblue')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(filepath, dpi=1200)
    plt.close()

def plot_scatter_chart(filepath, data_x, data_y, xlabel='Categories', ylabel='Values'):
    categories = data_x
    values = data_y

    title = f'{ylabel} w.r.t {xlabel}'
    plt.scatter(categories, values, color='skyblue')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(filepath, dpi=1200)
    plt.close()


def plot_curve(filepath, data_x, data_y, xlabel='Categories', ylabel='Values'):
    categories = data_x
    values = data_y

    title = f'{ylabel} w.r.t {xlabel}'
    plt.plot(categories, values, color='skyblue')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(filepath, dpi=1200)
    plt.close()


def append_to_csv_pandas(filename, data, header=False):
    df = pd.DataFrame(data)
    df.to_csv(filename, mode='a', header=header, index=False)


def full_split_training(config, result_folder, trained_models_folder, splits_to_compute):
    torch.cuda.empty_cache()
    gc.collect()
    torch.multiprocessing.set_sharing_strategy('file_system')
    multiprocessing.set_start_method('spawn', force=True)
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Set the current working directory to the script directory
    os.chdir(script_dir)

    num_workers = 0
    pin_memory = True


    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    batch_size = config.batch_size
    optimizer_type = config.optimizer_type
    if optimizer_type == OptimizerType.SGD:
        momentum = config.momentum
        weight_decay = config.weight_decay
    else:
        momentum = None


    # must be dividable by window size (7 in our case)
    train_transform, val_transform = get_augmentations(img_height=224, img_width=224)

    train_loader, val_loader, test_loader, color_map = get_loaders_camvid(
        1,
        train_transform,
        val_transform,
        num_workers,
        pin_memory
    )

    # acc_min = 70
    # acc_max = 80

    acc_min = [65, 41, 52, 32, 23, 50, 67, 66]
    acc_max = [79, 70, 71, 70, 70, 84, 84, 84]


    # data_max = 50000
    # data_min = 5000

    data_max = [602112, 188160, 78400, 19600, 5880, 19600, 78400, 188160]
    data_min = [200704, 12544, 3136, 784, 196, 784, 3136, 12544]

    gpu_num = 0
    device = f"cuda:{str(gpu_num)}" if torch.cuda.is_available() else "cpu"
    print('Device:', device)
    utils.create_device(device)
    device = utils.get_device()

    max_channel_bounds = [25, 30, 30, 30]



    resume_to_split = config.split_index
    for split_index in range(resume_to_split, len(splits_to_compute)):
        config.split_index = split_index
        print("training:", splits_to_compute[split_index])
        model_name = config.model_name if "{}" not in config.model_name else config.model_name.format( config.split_index)
        if not os.path.exists(os.path.join(result_folder, model_name)):
            os.makedirs(os.path.join(result_folder, model_name))
        if hasattr(config, 'splits'):
            config.splits = str(splits_to_compute[split_index])

        best_accs_graph = []
        best_scores_graph = []
        values_grid = [i for i in range(1, max_channel_bounds[config.split_index])]
        value_name = "Reduction channels"
        collect_dataset = True

        for value in values_grid:
            config.channels[config.split_index] = value
            model = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), channels=3, num_classes=32,
                                    head_dim=32, window_size=7, downscaling_factors=(4, 2, 2, 2),
                                    relative_pos_embedding=True, split_points=splits_to_compute[split_index],
                                    with_exit=False, split_channels=config.channels, split_after_patch_merging=True,
                                    reduction_factors=config.reduction_factors)



            model = model.to(device=device)
            model.set_strategy(CreateSplitDataset())
            train_split_dataset_samples, val_split_dataset_samples = [], []

            print("Loading model")
            loaded_checkpoint = model.load_checkpoint(path=f"{trained_models_folder}/{config.model_to_load}.pth.tar")

            if collect_dataset:
                data_collection_epochs = 5
                # setattr(config, 'data_collection_epochs', data_collection_epochs)

                for epoch in range(data_collection_epochs):
                    print(f"Run {epoch}/{data_collection_epochs} for training data")
                    start = time.time()
                    for x, _ in train_loader:
                        x = x.to(device=device)
                        x = model(x)
                        x = x.detach().cpu()
                        # for sample in x:
                        #     train_split_dataset_samples.append(sample)
                        train_split_dataset_samples.extend(x)
                    end = time.time()
                    print("Running:", end - start)

                for epoch in range(data_collection_epochs):
                    print(f"Run {epoch}/{data_collection_epochs} for validation data")
                    start = time.time()
                    for x, _ in val_loader:
                        x = x.to(device=device)
                        x = model(x)
                        x = x.to(device="cpu")
                        for sample in x:
                            val_split_dataset_samples.append(sample)

                    end = time.time()
                    print("Running:", end - start)

                torch.save(train_split_dataset_samples, f"{result_folder}/{model_name}/train_split_data.pth")
                torch.save(val_split_dataset_samples, f"{result_folder}/{model_name}/val_split_data.pth")
                exit(0)

            # train_split_dataset_samples = torch.load(f"{result_folder}/training_split_only/channels_50/split_{config.split_index}/train_split_data.pth")
            # val_split_dataset_samples = torch.load(f"{result_folder}/training_split_only/channels_50/split_{config.split_index}/val_split_data.pth")
            train_split_dataset_samples = torch.load(f"{result_folder}/{model_name}/train_split_data.pth")
            val_split_dataset_samples = torch.load(f"{result_folder}/{model_name}/train_split_data.pth")

            split_train_dataset = CustomSplitDataset(train_split_dataset_samples, augment=False)
            split_val_dataset = CustomSplitDataset(val_split_dataset_samples, augment=False)

            split_train_loader = DataLoader(split_train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
            split_val_loader = DataLoader(split_val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

            # model = model.to(device="cpu")
            # del model

            run_parallel_splits_only(split_train_loader, split_val_loader, model_name, result_folder, config)
            # run_training(multiprocess_model_num=1, config=config, train_loader=split_train_loader, val_loader=split_val_loader, model_name=model_name, result_folder=result_folder)
            swin_model = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), channels=3,
                                         num_classes=32, head_dim=32, window_size=7, downscaling_factors=(4, 2, 2, 2),
                                         relative_pos_embedding=True,
                                         split_points=utils.create_all_split_possibilities(send_result_back=False)[
                                             config.split_index], with_exit=False, split_channels=config.channels,
                                         split_after_patch_merging=True, reduction_factors=config.reduction_factors)
            swin_model = swin_model.to(device=device)
            val_scores, val_accs, val_mious = [], [], []
            datasize = -1
            for mpm_num in range(1, config.repeat_model + 1):
                val_acc, val_miou, datasize = (
                    evaluate_final_model(config, result_folder, model_name, mpm_num, swin_model, val_loader))

                val_accs.append(val_acc)
                val_mious.append(val_miou)

                if val_acc < acc_min[config.split_index]:
                    normalised_acc = 0
                elif val_acc > acc_max[config.split_index]:
                    normalised_acc = 1
                else:
                    normalised_acc = (val_acc - acc_min[config.split_index]) / (acc_max[config.split_index] - acc_min[config.split_index])

                if datasize[0] > data_max[config.split_index]:
                    normalised_datasize = 1
                elif datasize[0] < data_min[config.split_index]:
                    normalised_datasize = 0
                else:
                    normalised_datasize = (datasize[0] - data_min[config.split_index]) / (data_max[config.split_index] - data_min[config.split_index])
                normalised_score = (1 - normalised_datasize) + normalised_acc
                val_scores.append(normalised_score)

                create_each_model_csv(normalised_score, val_acc, val_miou, datasize, mpm_num, value, value_name
                                      ,filename=f'{result_folder}/{model_name}/results_each_run_{value_name}.csv')

            avg_acc = sum(val_accs) / len(val_accs)
            avg_miou = sum(val_mious) / len(val_mious)
            avg_score = sum(val_scores) / len(val_scores)
            create_total_csv(datasize, avg_acc, avg_miou, max(val_accs), max(val_mious), avg_score, max(val_scores), value,
                             value_name, filename=f'{result_folder}/{model_name}/results_total_{value_name}.csv')

            best_accs_graph.append(max(val_accs))
            best_scores_graph.append(max(val_scores))

            swin_model = swin_model.to("cpu")
            del swin_model, split_train_dataset, split_val_dataset

        plot_bar_chart(f"{result_folder}/{model_name}/acc_graph.png", values_grid, best_accs_graph, xlabel=value_name, ylabel="Best Accuracy")
        plot_bar_chart(f"{result_folder}/{model_name}/score_graph.png", values_grid, best_scores_graph, xlabel=value_name, ylabel="Best Score")
        plot_scatter_chart(f"{result_folder}/{model_name}/acc_graph.png", values_grid, best_accs_graph, xlabel=value_name, ylabel="Best Accuracy")
        plot_scatter_chart(f"{result_folder}/{model_name}/score_graph.png", values_grid, best_scores_graph, xlabel=value_name, ylabel="Best Score")
        plot_curve(f"{result_folder}/{model_name}/acc_graph_curve.png", values_grid, best_accs_graph, xlabel=value_name, ylabel="Best Accuracy")
        plot_curve(f"{result_folder}/{model_name}/score_graph_curve.png", values_grid, best_scores_graph, xlabel=value_name, ylabel="Best Score")
def create_each_model_csv(val_score, val_acc, val_miou, datasize, mpm_num, value, value_name, filename):
    data_row = {str(value_name): [value], 'Model number': mpm_num, 'Validation accuracy': val_acc,
                'Validation miou': val_miou, 'Validation score': val_score, 'Datasize': datasize}
    file_exists = os.path.isfile(filename)
    append_to_csv_pandas(filename, data_row, header=not file_exists)

    print(f"Data has been appended to {filename}")


def create_total_csv(datasize, avg_acc, avg_miou, best_acc, best_miou, avg_score, best_score, value, value_name, filename):
    data_row = {str(value_name): [value], 'Avg Accuracy': [avg_acc], 'Best Accuracy': [best_acc],
                'Avg Miou': [avg_miou], 'Best Miou': [best_miou], 'Avg Score': [avg_score], 'Best Score': [best_score], 'Datasize': datasize}

    file_exists = os.path.isfile(filename)
    append_to_csv_pandas(filename, data_row, header=not file_exists)

    print(f"Data has been appended to {filename}")


def run_parallel_splits_only(train_loader, val_loader, model_name, result_folder, config):
    model_nums = [i + 1 for i in range(config.repeat_model)]
    max_retries = 3
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_model = {executor.submit(run_with_retries, run_training, model_num, max_retries, train_loader,
                                           val_loader, model_name, result_folder, config)
                           : model_num for model_num in model_nums}

        for future in concurrent.futures.as_completed(future_to_model):
            model_num = future_to_model[future]
            try:
                result = future.result()
                print(f"Model{model_num} succeeded with result: {result}")
            except Exception as e:
                print(f"Model{model_num} failed after {max_retries} attempts with error: {e}")

        executor.shutdown(wait=True)
    # run_with_retries(run_training, model_num=1, max_retries=3, train_loader=train_loader, val_loader=val_loader,
    #                  model_name=model_name, result_folder=result_folder,  config=config)




def run_with_retries(task, model_num, max_retries, train_loader, val_loader, model_name, result_folder, config):
    # config = TrainingConfig()
    # config.resume_training = False
    # task(model_num, config, train_loader, val_loader, model_name, result_folder)
    for attempt in range(max_retries):
        try:
            result = task(model_num, config, train_loader, val_loader, model_name, result_folder)
            return result
        except Exception as e:
            config.resume_training = True
            print(f"{task.__name__} failed on attempt {attempt + 1} with error: {e}")
            if "CUDA out of memory" in str(e):
                print(f"Emptying cache")
                torch.cuda.empty_cache()  # Free up GPU memory
            if attempt == max_retries - 1:
                raise
            time.sleep(5)  # Wait a bit before retrying


def run_training(multiprocess_model_num: int, config, train_loader, val_loader, model_name, result_folder) -> str:
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(f"Running process for model {multiprocess_model_num}")

    gpu_num = 0
    device = f"cuda:{str(gpu_num)}" if torch.cuda.is_available() else "cpu"
    print('Device:', device)
    utils.create_device(device)
    device = utils.get_device()

    print(f"Training model {multiprocess_model_num} out of 3")
    train_loss_list = []
    val_loss_list = []
    learning_rates = []
    reduction_factors = config.reduction_factors
    # TODO: take directly from transformer - but should be static
    in_channels = [96, 192, 384, 768]
    split_channels = config.channels

    model = Autoencoder(in_channels=in_channels[config.split_index],
                        reduce_to_channels=split_channels[config.split_index],
                        reduction_factor=reduction_factors[config.split_index], index=config.split_index)


    model = model.to(device=device)
    model.set_strategy(TrainSplitOnly())
    # for early stopping
    min_val_loss = 100

    loss_func = model.get_loss_func()
    optimizer = model.initialize_optimizer(config.learning_rate, config.optimizer_type, momentum=config.momentum)
    scheduler = PolynomialLR(optimizer, total_iters=config.num_epochs, power=0.9)

    prev_epoch = 0
    # if config.resume_training and os.path.exists(f"{result_folder}/{model_name}/model{multiprocess_model_num}.pth.tar"):
    #     print("Loading model to continue training")
    #     loaded_checkpoint = model.load_checkpoint(path=f"{result_folder}/{model_name}/model{multiprocess_model_num}.pth.tar")
    #     train_loss_list = loaded_checkpoint["all_train_losses"]
    #     val_loss_list = loaded_checkpoint["all_val_losses"]
    #     learning_rates = loaded_checkpoint["learning_rates"]
    #     optimizer.load_state_dict(loaded_checkpoint["optimizer"])
    #     scheduler.load_state_dict(loaded_checkpoint["scheduler"])
    #     prev_epoch = loaded_checkpoint["epoch"]
    #
    # else:
    # evaluation before training
    print('Evaluation before training..')

    val_loss = model.check_batch_acc_miou_loss(val_loader, loss_func=loss_func)
    train_loss = model.check_batch_acc_miou_loss(train_loader, loss_func=loss_func)

    metric_dict = {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_loss_list": val_loss_list,
        "train_loss_list": train_loss_list
    }
    metric_dict = model.append_metric_lists(metric_dict, 0)
    train_loss = metric_dict.get("train_loss")
    val_loss = metric_dict.get("val_loss")
    train_loss_list = metric_dict.get("train_loss_list")
    val_loss_list = metric_dict.get("val_loss_list")

    checkpoint = {
        "epoch": 0,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "all_train_losses": train_loss_list,
        "all_val_losses": val_loss_list,
        "learning_rates": [config.learning_rate]
    }
    model.save_checkpoint(checkpoint, filename=model_name, folder=result_folder, save_model=False,
                          model_num=multiprocess_model_num)


    config.save_to_json(f"{result_folder}/{model_name}/training_config.json")
    for epoch in range(1, config.num_epochs + 1 - prev_epoch):
        print('Epoch:', prev_epoch + epoch, '/', config.num_epochs)

        _, _, train_loss = train(train_loader, model, optimizer, loss_func)
        learning_rates.append(optimizer.param_groups[0]["lr"])
        print('Calculating validation results')
        val_loss = model.check_batch_acc_miou_loss(val_loader, loss_func=loss_func)

        metric_dict = {
            "val_loss_list": val_loss_list,
            "train_loss_list": train_loss_list,
            "val_loss": val_loss,
            "train_loss": train_loss
        }

        metric_dict = model.append_metric_lists(metric_dict, epoch)

        scheduler.step()

        train_loss = metric_dict.get("train_loss")
        val_loss = metric_dict.get("val_loss")
        train_loss_list = metric_dict.get("train_loss_list")
        val_loss_list = metric_dict.get("val_loss_list")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": prev_epoch + epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "all_train_losses": train_loss_list,
            "all_val_losses": val_loss_list,
            "learning_rates": learning_rates,
        }
        # early stopping mechanism to prevent overtraining
        if sum(val_loss) < min_val_loss:
            model.save_checkpoint(checkpoint, filename=model_name, folder=result_folder, save_model=True,
                                  model_num=multiprocess_model_num)
            min_val_loss = sum(val_loss)
        else:
            model.save_checkpoint(checkpoint, filename=model_name, folder=result_folder, save_model=False,
                                  model_num=multiprocess_model_num)

        # model.draw_losses_graph_joint(train_acc_list, val_acc_list, f'{result_folder}/{model_name}/acc_graph{model_num}', "Accuracy")
        # model.draw_losses_graph_joint(train_miou_list, val_miou_list, f'{result_folder}/{model_name}/mious_graph{model_num}', "mIoU")
        # model.draw_losses_graph_joint(train_loss_list, val_loss_list,
        #                               f'{result_folder}/{model_name}/losses_graph{multiprocess_model_num}', "MSE Loss")
        # utils.draw_learning_rate(learning_rates, f'{result_folder}/{model_name}/lr_graph{model_num}')

    model = model.to("cpu")
    del model
    return f"Successful run for model {multiprocess_model_num}"

def evaluate_final_model(config, result_folder, model_name, multiprocess_model_num, swin_model, val_loader):

    # check final performance
    swin_model.set_strategy(TransformerTrainingStrategy.TrainingSplitsOneByOne())
    print("Loading model")
    swin_model.load_checkpoint(path=f"../trained_models/{config.model_to_load}.pth.tar")
    print("Loading split")
    split_checkpoint = torch.load(f"{result_folder}/{model_name}/model{multiprocess_model_num}.pth.tar"
                                  , map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    swin_model.load_state_dict(state_dict=split_checkpoint["state_dict"], strict=False)

    val_accuracy, val_miou, val_loss = swin_model.check_batch_acc_miou_loss(val_loader, loss_func=swin_model.get_loss_func())

    print("Validation acc:", val_accuracy)
    print("Validation miou", val_miou)


    logname = f"{result_folder}/{model_name}/log{str(multiprocess_model_num)}.txt"
    val_loss_array = [round(num, 3) for num in val_loss]
    val_acc_array = [round(num, 3) for num in val_accuracy]
    val_miou_array = [round(num, 3) for num in val_miou]

    with open(logname, 'a') as file:
        file.write(" validation loss: ")
        file.write(str(val_loss_array[0]))
        file.write(" validation accuracy: ")
        file.write(str(val_acc_array[0]))
        file.write(" validation mIoU: ")
        file.write(str(val_miou_array[0]))
        file.write(" transferred data size: ")
        file.write(str(swin_model.get_datasizes(mb=False)))
        file.write('\n')
    return val_acc_array[0], val_miou_array[0], swin_model.get_datasizes(mb=False)
