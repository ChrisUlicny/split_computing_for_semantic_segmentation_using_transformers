from bayes_opt import JSONLogger, BayesianOptimization, Events

import TransformerTrainingStrategy

import loaders
import os
import utils

from models.SwinTransformer import SwinTransformer
from TrainingConfig import TrainingConfig
import full_training

split_counter = 0
search_counter = 0
original_split_channels = [32, 64, 128, 256, 128, 64, 32]


def get_datasize_based_on_channels(channels, splits):
    model = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), channels=3, num_classes=32,
                            head_dim=32, window_size=7, downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True,
                            split_points=splits, with_exit=False, split_channels=channels, split_after_patch_merging=False,
                            reduction_factors=[4, 4, 2, 0, 0, 0, 2, 4])
    model.set_strategy(TransformerTrainingStrategy.TransformerInferenceStrategy())
    train_transform, val_transform = loaders.get_augmentations(img_height=224, img_width=224)
    train_loader, val_loader, test_loader, color_map = loaders.get_loaders_camvid(
        8,
        train_transform,
        val_transform,
        0,
        True
    )
    for idx, (image, mask) in enumerate(test_loader):
        image = image.to(device=utils.get_device())
        model(image)
        datasizes = model.get_datasizes()
        if len(datasizes) == 0:
            datasizes.append(0.0)
        print("Datasizes", datasizes)
        return datasizes


def find_hyperparameters(channels):
    global search_counter
    search_counter += 1
    all_split_possibilities = utils.create_all_split_possibilities(send_result_back=False)
    splits_to_compute = all_split_possibilities[split_counter]
    channels = int(round(channels))
    config = TrainingConfig()
    config.channels = channels
    config.model_name = f"search-{search_counter}"
    min_datasize = get_datasize_based_on_channels(5, splits_to_compute)
    max_datasize = get_datasize_based_on_channels(original_split_channels[split_counter-1], splits_to_compute)
    score = full_training.full_training(config, f"BO_results_one_by_one/splits{split_counter}", "../trained_models",
                                        [splits_to_compute], min_datasize=min_datasize, max_datasize=max_datasize)
    # print('Model score', score)
    return score


def do_bayes_opt():
    global split_counter, original_split_channels, search_counter
    search_counter = 0
    split_counter += 1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd, "BOLogs")):
        os.makedirs(os.path.join(cwd, "BOLogs"))

    min_channels = 5

    logger = JSONLogger(path=os.path.join(cwd, f'BOLogs_one_by_one/splits_hyperparameters{split_counter}.json'))
    search_space = {'channels': (min_channels, original_split_channels[split_counter-1])}
    bo = BayesianOptimization(f=find_hyperparameters, pbounds=search_space, random_state=42, allow_duplicate_points=True,
                              verbose=2)

    bo.subscribe(Events.OPTIMIZATION_STEP, logger)
    bo.probe(
        params={'channels': min_channels},
        lazy=True,
    )
    bo.probe(
        params={'channels': original_split_channels[split_counter-1]},
        lazy=True,
    )

    # load_logs(bo, logs=['BOLogs/logs_new.json'])
    bo.maximize(init_points=2, n_iter=10)
    print(bo.max)

