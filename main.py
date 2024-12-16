import argparse
import warnings

import torch
import os

import utils.utils as utils
from TrainingConfig import TrainingConfig
from full_training import full_training
from misc.optimize_function import do_bayes_opt
from utils.simulation_utils import collect_layer_data
import utils.graph_utils as split_utils



if __name__ == "__main__":

    # Set the environment variable for max_split_size_mb to manage fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    config = TrainingConfig()
    gpu_num = config.gpu_num
    device = f"cuda:{str(gpu_num)}" if torch.cuda.is_available() else "cpu"
    print('Device:', device)
    utils.create_device(device)

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # config = TrainingConfig()

    parser = argparse.ArgumentParser()
    # Add the argument
    parser.add_argument('--bayes', type=bool, help='If true, use Bayesian ')
    parser.add_argument('--simulation', type=bool, help='If true, run simulation')
    parser.add_argument('--get_data', type=bool, help='If true, get layer data')
    parser.add_argument('--repair_score', type=bool)
    parser.add_argument('--repair_datasizes', type=bool)
    parser.add_argument('--create_graphs', type=bool)
    # Parse the arguments
    args = parser.parse_args()


    if args.bayes:
        print("Running bayes optimization")
        for i in range(0, 7):
            do_bayes_opt()
    elif args.simulation:
        config = TrainingConfig()
        all_split_possibilities = utils.create_all_split_possibilities(send_result_back=False)
        # split_utils.plot_bar_chart("../simulations_final/accuracies.png")
        # exit(0)
        device_ips = [1.1e12, 4.1e12, 7.1e12]
        resource_blocks = [2000, 1000, 4000]

        for i in range(len(device_ips)):
            for j in range(len(resource_blocks)):
                # Create a copy of the original config dictionary
                # modified_config = copy.deepcopy(simulation_config_dict)
                # modified_config["vehicles"][0]["ips"] = device_ips[i]
                # modified_config["base_stations"]["resource_blocks"] = resource_blocks[j]
                # modified_config["result_dir"] = ("/results/culicny/simulations_tests/" + str(device_ips[i]/1e12) + "_" + str(resource_blocks[j])
                #                                  + "/split{}/channels{}/run_{}/")
                # start_idx = config.split_index
                # for split_index in range(start_idx, len(all_split_possibilities)):
                #     split_channels = [-1, 3, 5, 5, 6, 5, 6, 3, -1]
                #     print(f"Running simulation for split number {split_index} using {split_channels[split_index]} channels!")
                #     execute_whole_simulation(SimulationConfig(**modified_config), split_index, split_channels[split_index])
                # split_utils.create_simulation_graph_total("../simulations_final/" + str(device_ips[i]/1e12) + "_" + str(resource_blocks[j]), num_splits= 9, num_runs=10, num_cars=10)
                split_utils.create_simulation_graph_separate("../simulations_final/simulations_final/" + str(device_ips[i]/1e12) + "_" + str(resource_blocks[j]), num_splits= 9, num_runs=10, num_cars=10, fontsize=18)

    elif args.get_data:
        print("Getting layer data")
        collect_layer_data(save_to="../layer_data")

    elif args.repair_datasizes:
        print("Repairing datasizes")
        for split_index in range(0, 8):
            split_utils.repair_datasize(f"../csvs/split{split_index}/results_total_Split channels.csv", split_index)

    elif args.repair_score:
        print("Repairing score ")
        # split_utils.repair_csv(f"../channel_search_old", f"../csvs/split{4}/results_total_Split channels.csv")
        for split_index in range(1, 5):
            split_utils.repair_score(f"../csvs/split{split_index}/results_total_Split channels.csv", split_index)

    elif args.create_graphs:
        split_utils.generate_acc_graph_from_json(fontsize=18)
        # graph_utils.img_to_grayscale("../images_paper/accuracies.png")
        split_utils.plot_bar_chart("../simulations_final/accuracies.png", fontsize=18)
        for split_index in range(0, 8):
            # split_utils.create_graph_score_split(f"../csvs/split{split_index}/results_total_Split channels.csv", split_index)
            # split_utils.create_graph_acc_split(f"../csvs/split{split_index}/results_total_Split channels.csv",
            #                                      split_index)
            split_utils.create_combined_graph(f"../csvs/split{split_index}/results_total_Split channels.csv", split_index, fontsize=18)
            # split_utils.create_three_combined_graph(f"../csvs/split{split_index}/results_total_Split channels.csv", split_index)
        # # split_utils.create_simulation_graph_total(f"../simulation_results", num_splits= 9, num_runs=10, num_cars=10)
        # # split_utils.create_simulation_graph_separate(f"../simulation_results", num_splits= 9, num_runs=10, num_cars=10)

    else:
        print("Starting full training")
        splits_to_compute = utils.create_all_split_possibilities(send_result_back=False)
        # splits_to_compute = splits_to_compute[:-5]
        splits_to_compute = [splits_to_compute[-1]]
        # split_training.full_split_training(config=config, result_folder="/results/culicny/models",
        #                                    trained_models_folder="../trained_models", splits_to_compute=splits_to_compute)
        # run_parallel()
        # run_with_retries(run_training, None, max_retries=1)

        full_training(config, result_folder="../results/models",
                      trained_models_folder="../trained_models", splits_to_compute=splits_to_compute, mp_model_num=1)



