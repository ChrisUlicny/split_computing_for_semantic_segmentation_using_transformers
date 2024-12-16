import json
import os
import pickle
import osmnx as ox
from fvcore.nn import FlopCountAnalysis
import time
import TransformerTrainingStrategy
import loaders
import utils
from models.SwinTransformer import SwinTransformer


def get_flops_and_size(model):
    model.set_strategy(TransformerTrainingStrategy.TransformerInferenceStrategy())
    model = model.to(utils.get_device())
    train_transform, val_transform = loaders.get_augmentations(img_height=224, img_width=224)
    train_loader, val_loader, test_loader, color_map = loaders.get_loaders_camvid(
        8,
        train_transform,
        val_transform,
        0,
        True
    )
    mobile_flops, server_flops, prediction, mask = None, None, None, None
    datasizes = [None]
    for idx, (image, mask) in enumerate(test_loader):
        image = image.to(device=utils.get_device())
        mask = mask.to(device=utils.get_device())
        flops = FlopCountAnalysis(model, image)
        mobile_flops, server_flops = utils.calculate_before_and_after_flops(flops.by_module(), model.get_split_point())

        print("total Flops:", flops.total())
        print("mobile Flops:", mobile_flops)
        print("server Flops:", server_flops)
        start_time = time.time()
        prediction = model(image)
        end_time = time.time()
        datasizes = model.get_datasizes(mb=False)
        if len(datasizes) == 0:
            datasizes.append(0.0)
        print("Data size in bytes:", datasizes[0])
        print("Model execution time:", end_time - start_time)
        break

    # return mobile_flops, server_flops, datasizes[0], prediction, mask
    return mobile_flops, server_flops, datasizes[0]



def generate_manhattan_graph():
    # Define a smaller bounding box for demonstration purposes
    north, south, east, west = 40.82, 40.78, -73.97, -74.01
    #
    # center_latitude = 40
    # center_longitude = 40
    # offset = 0.04  # This is a rough estimate to cover a reasonable area around the center
    #
    # north = center_latitude + offset
    # south = center_latitude - offset
    # east = center_longitude + offset
    # west = center_longitude - offset

    G = ox.graph_from_bbox(north, south, east, west, network_type='drive', simplify=False)
    ox.plot_graph(G)
    # G = ox.truncate.truncate_graph_grid(G, north, south, east, west, rows, cols, truncate_by_edge=True)

    # Save the graph in .pkl format
    with open('graph_center_40.pkl', 'wb') as f:
        pickle.dump(G, f)


def collect_layer_data(save_to):
    print("Getting layer data")
    if os.path.exists(f"{save_to}/transformer_splits.json"):
        with open(f"{save_to}/transformer_splits.json", 'r') as json_file:
            results = json.load(json_file)
    else:
        if not os.path.exists(save_to):
            os.mkdir(save_to)
        results = {}
    all_split_possibilities = utils.create_all_split_possibilities(send_result_back=False)
    print(all_split_possibilities)
    split_channels = [-1, 3, 5, 5, 6, 5, 6, 3, -1]
    for split_idx in range(0, len(all_split_possibilities)):
        # print(split_idx)
        print(all_split_possibilities[split_idx])
        model = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), channels=3, num_classes=32,
                                head_dim=32, window_size=7, downscaling_factors=(4, 2, 2, 2),
                                relative_pos_embedding=True, split_points=all_split_possibilities[split_idx],
                                with_exit=False, split_channels=split_channels, split_after_patch_merging=False,
                                reduction_factors=[0, 0, 0, 0, 0, 0, 0, 0])
        mobile_flops, server_flops, size = get_flops_and_size(model=model)

        # Ensure the top-level key for split_idx exists
        if str(split_idx) not in results:
            results[str(split_idx)] = {}

        # Ensure the nested key for split_channels exists within the current split_idx
        if str(split_channels[split_idx]) not in results[str(split_idx)]:
            results[str(split_idx)][str(split_channels[split_idx])] = {}

        # Add the results to the appropriate location
        results[str(split_idx)][str(split_channels[split_idx])] = {
            "mobile_flops": mobile_flops,
            "server_flops": server_flops,
            "size": size
        }

        with open(f"{save_to}/transformer_splits.json", 'w') as json_file:
            json.dump(results, json_file, indent=4)
