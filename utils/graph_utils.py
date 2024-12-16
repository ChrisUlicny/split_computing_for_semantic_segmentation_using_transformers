import json

import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mp



from PIL import Image

def img_to_grayscale(filepath):
    # Load the image
    image = Image.open(filepath)
    # Convert the image to grayscale
    gray_image = image.convert('L')
    # Save the grayscale image
    filepath = filepath.replace('.png', '_g.png')
    gray_image.save(filepath)


def get_split_index(split_array):
    for index, value in enumerate(split_array):
        if value:
            return index
    # if all False
    return 8

def get_used_memory(tensor):
    # print("shape of tensor:", tensor.shape)
    used_memory_bytes = (torch.numel(tensor) * tensor.element_size())
    # print(f"{used_memory_bytes} = {torch.numel(tensor)}  x {tensor.element_size()} ")
                   #+ sys.getsizeof(tensor))
    return used_memory_bytes


def create_graph_score_split(filepath, split_index):
    df = pd.read_csv(filepath)
    avg_scores = df['Avg Score']
    split_channels = df['Split channels']

    plt.bar(split_channels, avg_scores, color='skyblue')

    plt.xlabel('Channel-reduction value c')
    plt.ylabel(r"Score $\alpha_{s}(c)$")

    plt.savefig(f"../csvs/split{split_index}/split_score.png", dpi=1200)
    plt.close()


def create_graph_acc_split(filepath, split_index):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    plt.rcParams['axes.unicode_minus'] = False

    df = pd.read_csv(filepath)
    avg_accs = df['Avg Accuracy']
    split_channels = df['Split channels']

    plt.plot(split_channels, avg_accs, color='red')

    plt.xlabel('Channel-reduction value c')
    plt.ylabel(r"Accuracy $\mu_{s}(c)$")

    plt.savefig(f"../csvs/split{split_index}/split_acc.png", dpi=1200)
    plt.close()


def create_combined_graph(filepath, split_index, fontsize):
    mp.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.gca().set_axisbelow(True)

    df = pd.read_csv(filepath)
    split_channels = df['Split channels']
    avg_scores = df['Avg Score']
    avg_accs = df['Avg Accuracy']

    fig, ax1 = plt.subplots()
    # Set tick size for x-axis
    ax1.tick_params(axis='x', labelsize=fontsize)
    color = '#4083c7'
    ax1.set_xlabel('Channel-reduction value c', fontsize=fontsize)
    ax1.set_ylabel(r"Score $\alpha_{s}(c)$", fontsize=fontsize, color=color)
    ax1.bar(split_channels, avg_scores, color="skyblue", zorder=15)
    # Set tick size for left y-axis
    ax1.tick_params(axis='y', labelcolor=color, labelsize=fontsize)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = '#ad0707'
    ax2.set_ylabel(r"Accuracy $\mu_{s}(c)$ $\left[\%\right]$", fontsize=fontsize, color=color)  # we already handled the x-label with ax1
    ax2.plot(split_channels, avg_accs, color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=fontsize)

    # Add grid lines for x-axis and primary y-axis
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)

    # Add grid lines for secondary y-axis
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"../csvs/split{split_index}/combined_split.png", dpi=1200)
    img_to_grayscale(f"../csvs/split{split_index}/combined_split.png")
    plt.close()


def create_three_combined_graph(filepath, split_index):
    df = pd.read_csv(filepath)
    split_channels = df['Split channels']
    avg_scores = df['Avg Score']
    avg_accs = df['Avg Accuracy']
    data_sizes = df['Datasize']  # Assuming this is the third y-value you want to plot

    fig, ax1 = plt.subplots()

    # Plotting the average scores
    color = 'tab:blue'
    ax1.set_xlabel('Channel-reduction value c')
    ax1.set_ylabel(r"Score $\alpha_{s}(c)$", color=color)
    ax1.bar(split_channels, avg_scores, color='skyblue')
    ax1.tick_params(axis='y', labelcolor=color)

    # Creating the second y-axis for accuracy and precision
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel(r"Accuracy $\mu_{s}(c)$ (%)", color=color)
    ax2.plot(split_channels, avg_accs, color=color)
    ax2.tick_params(axis='y', labelcolor=color)


    # # Creating the third y-axis for precision
    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.spines["right"].set_position(("outward", 60))  # Move the spine outwards by 60 points
    ax3.set_ylabel(r"Data size $D_{s}(c)$ (B)", color=color)
    ax3.plot(split_channels, data_sizes, color=color, linestyle='dashed')
    ax3.tick_params(axis='y', labelcolor=color)

    # # Creating the third y-axis for precision
    # ax3 = ax1.twinx()
    # color_prec = 'tab:green'
    # ax3.set_ylabel(r"Data size $D_{s}(c)$", color=color_prec)
    # ax3.plot(split_channels, data_sizes, color=color_prec, linestyle='dashed', label='Data size')
    # ax3.tick_params(axis='y', labelcolor=color_prec)
    #
    # # Adjusting the position of the third y-axis
    # ax3.yaxis.set_label_position('right')

    plt.grid(linestyle='--', linewidth=0.5)
    fig.tight_layout()  # To prevent clipping of the right y-label
    plt.savefig(f"../csvs/split{split_index}/combined_split_three.png", dpi=1200)
    plt.close()



def plot_bar_chart(filepath, fontsize):
    mp.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    categories =  ['ECS-Only', 'Split1', 'Split2', 'Split3', 'Split4', 'Split5', 'Split6', 'Split7',
              'CAV-Only']
    values = [83.98, 77.521, 82.251, 81.048, 81.111, 79.214, 81.966, 80.954, 83.98]
    # plt.figure(figsize=(16, 12))
    plt.ylim(0, 100)
    plt.bar(categories, values, label='Accuracy', color='#eb6e67', zorder=5)
    plt.xlabel('Split point', fontsize=fontsize)
    plt.ylabel(r'Accuracy $\mu_{s}(c)$ [\%]', fontsize=fontsize)
    plt.xticks(rotation=40, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # Add grid behind the bars
    plt.grid(linestyle='--', linewidth=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(filepath, dpi=1200)
    img_to_grayscale(filepath)
    plt.close()

def create_simulation_graph_separate(simulation_results_file, num_runs, num_cars, num_splits, fontsize):
    mp.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.gca().set_axisbelow(True)
    channels = [-1, 3, 5, 5, 6, 5, 6, 3, -1]
    # accuracies = [83.98, 64.66, 70.75, 65.03, 61.39, 79.21, 81.84, 81.42, 83.98]
    all_data_server = []
    all_data_device = []
    all_data_data = []
    for split in range(num_splits):
        data_lat_splitwise = []
        server_lat_splitwise = []
        device_lat_splitwise = []
        for run in range(num_runs):
            for car in range(num_cars):
                filepath = f"{simulation_results_file}/split{split}/channels{channels[split]}/run_{run}/car{car}/data/data_latency.txt"
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    numbers = [float(line.strip()) for line in lines]
                    data_lat_splitwise.extend(numbers)

                filepath = f"{simulation_results_file}/split{split}/channels{channels[split]}/run_{run}/car{car}/data/device_latency.txt"
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    numbers = [float(line.strip()) for line in lines]
                    device_lat_splitwise.extend(numbers)

                filepath = f"{simulation_results_file}/split{split}/channels{channels[split]}/run_{run}/car{car}/data/server_latency.txt"
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    numbers = [float(line.strip()) for line in lines]
                    server_lat_splitwise.extend(numbers)

        server_split_avg = sum(server_lat_splitwise) / len(server_lat_splitwise)
        device_split_avg = sum(device_lat_splitwise) / len(device_lat_splitwise)
        data_split_avg = sum(data_lat_splitwise) / len(data_lat_splitwise)
        all_data_server.append(server_split_avg)
        all_data_device.append(device_split_avg)
        all_data_data.append(data_split_avg)

    splits = ['ECS-Only', 'Split1', 'Split2', 'Split3', 'Split4', 'Split5', 'Split6', 'Split7',
              'CAV-Only']

    all_data_device = [item*1000 for item in all_data_device]
    all_data_server = [item*1000 for item in all_data_server]
    all_data_data = [item*1000 for item in all_data_data]

    t_total = [sum(x) for x in zip(all_data_device, all_data_server, all_data_data)]


    with open(f'{simulation_results_file}/latency_values.txt', 'w') as f:
        f.write("Split Point, Data Latency (ms), Device Latency (ms), Server Latency (ms), Total Latency (ms)\n")
        for i, split in enumerate(splits):
            f.write(f"{split}, {all_data_data[i]:.2f}, {all_data_device[i]:.2f}, {all_data_server[i]:.2f}, {t_total[i]:.2f}\n")


    # Create a stacked bar chart
    bar_width = 0.5
    index = np.arange(len(t_total))
    fig, ax1 = plt.subplots(dpi=1200)


    bar_t_uplink = r'Data offloading latency $t^{uplink}_z$'
    bar_t_cav = r'Computation latency on CAV $t^{CAV}_{z, v}$'
    bar_t_bs = r'Computation latency on ECS $t^{ECS}_z$'


    # Plot latency
    ax1.bar(index, all_data_device, bar_width, label=bar_t_cav, color='#000080', zorder=3)
    ax1.bar(index, all_data_server, bar_width, bottom=all_data_device, label=bar_t_bs, color='#21b021', zorder=3)
    ax1.bar(index, all_data_data, bar_width, bottom=np.array(all_data_device) + np.array(all_data_server), label=bar_t_uplink,
            color='#b3cde3', zorder=3)
    ax1.set_xlabel('Split point', fontsize=fontsize)
    ax1.set_xticks(index)
    ax1.set_xticklabels(splits, rotation=45)
    ax1.set_ylabel(r'Average latency $t^{total}_z$' + " [ms]", fontsize=fontsize)
    ax1.legend(loc='upper left', fontsize=17)

    # Set y-axis limit for latency
    ax1.set_ylim(0, 17)

    # # Create second y-axis for accuracy
    # ax2 = ax1.twinx()
    # # Plot accuracy as bars
    # ax2.bar(index + bar_width, accuracies, bar_width, label='Accuracy', color='red', alpha=0.6)
    # ax2.set_ylabel(u'\u03bc (%)')
    # ax2.legend(loc='upper right')
    #
    # # Set y-axis limit for accuracy
    # ax2.set_ylim(0, max(accuracies) * 1.5)

    plt.grid(linestyle='--', linewidth=0.5, zorder=0)
    # plt.figure(figsize=(8, 6))
    # Add and customize grid
    # plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f'{simulation_results_file}/latency_accuracy_graph_separate_ms.png')
    img_to_grayscale(f'{simulation_results_file}/latency_accuracy_graph_separate_ms.png')
    plt.close()



def create_simulation_graph_total(simulation_results_file, num_runs, num_cars, num_splits):
    channels = [-1, 3, 5, 5, 6, 5, 6, 3, -1]
    all_data = []
    for split in range(num_splits):
        all_data_splitwise = []
        for run in range(num_runs):
            for car in range(num_cars):
                filepath = f"{simulation_results_file}/split{split}/channels{channels[split]}/run_{run}/car{car}/data/latency.txt"
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    numbers = [float(line.strip()) for line in lines]
                    all_data_splitwise.extend(numbers)
        split_avg = sum(all_data_splitwise) / len(all_data_splitwise)
        print(split_avg)
        all_data.append(split_avg)

    splits = ['On Cloud', 'Encoder1', 'Encoder2', 'Encoder3', 'Encoder4', 'Decoder1', 'Decoder2', 'Decoder3',
              'On Mobile']

    # Create bar plot
    plt.figure(dpi=1200)
    plt.bar(splits, all_data, color='skyblue')

    # Add labels and title
    plt.xticks(rotation=45)
    plt.xlabel('Split after layer')
    plt.ylabel('Total Latency (s)')

    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{simulation_results_file}/latency_graph.png')
    plt.close()




def generate_split_graphs_separate(cloud_flops, mobile_flops, t_uplink_avg, t_downlink_avg, file):
    t_data_transfer = []

    # Nvidia GeForce 40 series (in FLOPS per second)
    avg_cloud_computational_power = 1.32 * pow(10, 15)

    # iPhone13 (in FLOPS per second)
    avg_mobile_computational_power = 1.2 * pow(10, 12)


    # Data for different splits and their total latency
    splits = ['On Cloud', 'Encoder1', 'Encoder2', 'Encoder3', 'Encoder4', 'Decoder1', 'Decoder2', 'Decoder3',
              'On Mobile']

    # for i in range(0, len(cloud_flops)):
    #     t_data_transfer.append(t_uplink_avg[i] + t_downlink_avg[i])

    t_data_transfer = [sum(x) for x in zip(t_uplink_avg, t_downlink_avg)]

    t_mobile = [i/avg_mobile_computational_power for i in mobile_flops]
    print("Mobile processing:", t_mobile)
    t_cloud = [i/avg_cloud_computational_power for i in cloud_flops]
    print("Cloud processing:", t_cloud)
    print("Data processing:", t_data_transfer)

    # Calculate the sum of each set of values
    t_total = [sum(x) for x in zip(t_mobile, t_cloud, t_data_transfer)]

    # Create a stacked bar chart
    bar_width = 0.35
    index = np.arange(len(t_total))
    plt.figure(dpi=600)
    plt.bar(index, t_mobile, bar_width, label='Mobile', color='skyblue')
    plt.bar(index, t_cloud, bar_width, bottom=t_mobile, label='Cloud', color='#fabc37')
    plt.bar(index, t_data_transfer, bar_width, bottom=np.array(t_mobile) + np.array(t_cloud), label='Data', color='#54f066')
    # plt.figure(figsize=(10, 6))


    plt.xlabel('Split Location')
    plt.xticks(rotation=45)
    plt.ylabel('Total Time (s)')
    plt.title('Total Time w.r.t Split Location')
    plt.xticks(index, splits)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{file}_c.png')
    plt.close()



def repair_score(filepath, split_index):
    # Define the ranges for normalization
    acc_min = [65, 48.188, 60.864, 54.327, 55.867, 50.654, 67.842, 66.907]
    acc_max = [79, 77.417, 81.094, 80.25, 81.527, 83.78, 83.776, 83.969]
    data_max = [602112, 188160, 78400, 19600, 5880, 19600, 78400, 188160]
    data_min = [200704, 12544, 3136, 784, 196, 784, 3136, 12544]

    # Load the CSV file
    df = pd.read_csv(filepath)

    # Initialize columns for updated scores
    avg_scores = []
    best_scores = []

    # Loop through each row to compute normalized scores
    for index, row in df.iterrows():
        avg_val_acc = row['Avg Accuracy']
        best_val_acc = row['Best Accuracy']
        datasize = row["Datasize"]

        # Normalize accuracy
        if avg_val_acc < acc_min[split_index]:
            normalised_avg_acc = 0
        elif avg_val_acc > acc_max[split_index]:
            normalised_avg_acc = 1
        else:
            normalised_avg_acc = (avg_val_acc - acc_min[split_index]) / (acc_max[split_index] - acc_min[split_index])

        if best_val_acc < acc_min[split_index]:
            normalised_best_acc = 0
        elif best_val_acc > acc_max[split_index]:
            normalised_best_acc = 1
        else:
            normalised_best_acc = (best_val_acc - acc_min[split_index]) / (acc_max[split_index] - acc_min[split_index])

        # Normalize datasize
        if datasize > data_max[split_index]:
            normalised_datasize = 1
        elif datasize < data_min[split_index]:
            normalised_datasize = 0
        else:
            normalised_datasize = (datasize - data_min[split_index]) / (data_max[split_index] - data_min[split_index])

        # Calculate normalized scores
        normalised_avg_score = (1 - normalised_datasize) + normalised_avg_acc
        normalised_best_score = (1 - normalised_datasize) + normalised_best_acc

        # Append the scores
        avg_scores.append(normalised_avg_score)
        best_scores.append(normalised_best_score)

    # Update the DataFrame
    df['Avg Score'] = avg_scores
    df['Best Score'] = best_scores

    # Save the modified DataFrame back to a new CSV file
    df.to_csv(filepath, index=False)

    print(f"Modified CSV saved to {filepath}")




def generate_acc_graph_from_json(fontsize):
    mp.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    split = 2
    channel = 4
    val_acc_list = []
    train_acc_list = []
    open_file = f"../../split_2_4_channels_log.json"
    with open(open_file, 'r') as file:
        split_data = json.load(file)
        for record in split_data:
            for exit_data in record['exits']:
                val_accuracy = exit_data['validation_accuracy']
                train_accuracy = exit_data['training_accuracy']
                val_acc_list.append(val_accuracy)
                train_acc_list.append(train_accuracy)

    scale = 0
    colors = [['m', '--g']]
    labels = ['Validation accuracy', 'Training accuracy']
    plt.figure(dpi=1200)
    for i in range(1):
        plt.plot(range(scale, len(val_acc_list) + scale), val_acc_list, colors[i][0], label=labels[i * 2])
        plt.plot(range(scale, len(train_acc_list) + scale), train_acc_list, colors[i][1], label=labels[i * 2 + 1])
    plt.legend(loc='lower right', fontsize=fontsize)
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel(r'Accuracy $\mu_{s}(c)$ [\%]', fontsize=fontsize)
    plt.xticks(range(scale, len(train_acc_list) + scale, 10), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.grid(linestyle='--', linewidth=0.5)
    plt.savefig(f'../split2_training.png')
    img_to_grayscale(f'../../split2_training.png')
    plt.close()



def repair_csv(filepath_files, filepath_csv):
    # Load the CSV file
    df = pd.read_csv(filepath_csv)
    # Open and read the JSON file
    highest_val_accuracies_per_channel = []
    for split in range(4, 5):
        for channel in range(1, 30):
            open_file = f"{filepath_files}/channels_{channel}/split_{split}/log1.json"
            with open(open_file, 'r') as file:
                split_data = json.load(file)
                highest_val_accuracy = -1
                # Iterate through the data to find the highest validation accuracy
                for record in split_data:
                    for exit_data in record['exits']:
                        val_accuracy = exit_data['validation_accuracy']
                        if val_accuracy > highest_val_accuracy:
                            highest_val_accuracy = val_accuracy
                highest_val_accuracies_per_channel.append(highest_val_accuracy)


    # # Loop through each row to compute normalized scores
    # for index, row in df.iterrows():
    #     print(index)
    #     print(row['Avg Accuracy'])
    #     row['Avg Accuracy'] = highest_val_accuracies_per_channel[index]
    #     print(row['Avg Accuracy'])


    count_of_channels_to_change = len(df['Avg Accuracy'] )
    highest_val_accuracies_per_channel = highest_val_accuracies_per_channel[:count_of_channels_to_change]
    # Update the DataFrame
    df['Avg Accuracy'] = highest_val_accuracies_per_channel

    # Save the modified DataFrame back to a new CSV file
    df.to_csv(filepath_csv, index=False)

    print(f"Modified CSV saved to {filepath_csv}")


def repair_datasize(filepath, split_index):

    # Load the CSV file
    df = pd.read_csv(filepath)
    # Initialize columns for updated scores

    # Loop through each row to compute normalized scores
    new_datasizes = []
    spatial_dims = [224, 56, 28, 14, 7, 14, 28, 56]
    split_channels = 1
    for index, row in df.iterrows():


        split_channels = row["Split channels"]
        new_datasizes.append(spatial_dims[split_index] * spatial_dims[split_index] * split_channels * 4)


    # Update the DataFrame
    df['Datasize'] = new_datasizes


    # Save the modified DataFrame back to a new CSV file
    df.to_csv(filepath, index=False)

    print(f"Modified CSV saved to {filepath}")


# def get_datasize_based_on_channels(channels, splits):
#     model = SwinTransformer.SwinTransformer(
#         hidden_dim=96,
#         layers=(2, 2, 6, 2),
#         heads=(3, 6, 12, 24),
#         channels=3,
#         num_classes=32,
#         head_dim=32,
#         window_size=7,
#         downscaling_factors=(4, 2, 2, 2),
#         relative_pos_embedding=True,
#         split_points=splits,
#         with_exit=False,
#         split_channels=channels
#     )
#     model.set_strategy(TransformerTrainingStrategy.TransformerInferenceStrategy())
#     train_transform, val_transform = loaders.get_augmentations(img_height=224, img_width=224)
#     train_loader, val_loader, test_loader, color_map = loaders.get_loaders_camvid(
#         8,
#         train_transform,
#         val_transform,
#         0,
#         True
#     )
#     for idx, (image, mask) in enumerate(test_loader):
#         image = image.to(device=get_device())
#         model(image)
#         datasizes = model.get_datasizes()
#         if len(datasizes) == 0:
#             datasizes.append(0.0)
#         print("Datasizes", datasizes)
#         return datasizes

