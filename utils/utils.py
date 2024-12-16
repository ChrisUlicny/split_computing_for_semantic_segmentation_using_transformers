import re
from enum import Enum
import copy
from math import floor

from datasets import *
from PIL import Image
import torch.nn.modules.module

import json

import metrics

global device

# Mbps
class NetworkType(Enum):

    WIFI = 18.88
    THREE_G = 1.1
    FOUR_G = 5.85



def save_checkpoint_joint(state, model_name, folder, save_model, model_num=0, exits_num=3):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    cwd = os.getcwd()
    print("Saving checkpoint")
    if not os.path.exists(os.path.join(cwd, folder)):
        print('Creating directory', os.makedirs(os.path.join(cwd, folder), exist_ok=True))
        try:
            os.makedirs(os.path.join(cwd, folder), exist_ok=True)
        except Exception as e:
            print(f"Error creating directory: {e}")
    if not os.path.exists(os.path.join(cwd, f"{folder}/{model_name}")):
        print('Creating directory', os.path.join(cwd, f"{folder}/{model_name}"))
        try:
            os.makedirs(os.path.join(cwd, f"{folder}/{model_name}"))
        except Exception as e:
            print(f"Error creating directory: {e}")
    filename = os.path.join(cwd, f"{folder}/{model_name}/model{str(model_num)}.pth.tar")
    logname = os.path.join(cwd, f"{folder}/{model_name}/log{str(model_num)}.txt")
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # if not os.path.exists(f"{folder}/{model_name}"):
    #     os.makedirs(f"{folder}/{model_name}")
    # filename = f"{folder}/{model_name}/model{str(model_num)}.pth.tar"
    # logname = f"{folder}/{model_name}/log{str(model_num)}.txt"
    train_loss_array = [round(num, 3) for num in state["train_loss"]]
    train_acc_array = [round(num, 3) for num in state["train_accuracy"]]
    train_miou_array = [round(num, 3) for num in state["train_miou"]]
    val_loss_array = [round(num, 3) for num in state["val_loss"]]
    val_acc_array = [round(num, 3) for num in state["val_accuracy"]]
    val_miou_array = [round(num, 3) for num in state["val_miou"]]
    if save_model:
        print("Saving model weights")
        torch.save(state, filename)
    else:
        print("Not saving model weights")
    with open(logname, 'a') as file:
        file.write("epoch: ")
        file.write(str(state["epoch"]))
        file.write(" lr: ")
        file.write(str(state["learning_rates"][-1]))
        file.write('\n')
        for i in range(0, exits_num):
            file.write(f"Exit number {exits_num-i}: \n")
            file.write(" training loss: ")
            file.write(str(train_loss_array[i]))
            file.write(" training accuracy: ")
            file.write(str(train_acc_array[i]))
            file.write(" training mIoU: ")
            file.write(str(train_miou_array[i]))
            file.write(" validation loss: ")
            file.write(str(val_loss_array[i]))
            file.write(" validation accuracy: ")
            file.write(str(val_acc_array[i]))
            file.write(" validation mIoU: ")
            file.write(str(val_miou_array[i]))
            file.write('\n')
        file.write('\n')


def save_checkpoint_split_json(state, filepath, save_model, exits_num, model_num=0):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    cwd = os.getcwd()
    print("Saving checkpoint")


    if not os.path.exists(filepath):
        print('Creating directory', filepath)
        try:
            os.makedirs(filepath)
        except Exception as e:
            print(f"Error creating directory: {e}")

    filename = os.path.join(cwd, f"{filepath}/model{str(model_num)}.pth.tar")
    logname = os.path.join(cwd, f"{filepath}/log{str(model_num)}.json")

    train_loss_array = [round(num, 3) for num in state["train_loss"]]
    train_acc_array = [round(num, 3) for num in state["train_accuracy"]]
    train_miou_array = [round(num, 3) for num in state["train_miou"]]
    val_loss_array = [round(num, 3) for num in state["val_loss"]]
    val_acc_array = [round(num, 3) for num in state["val_accuracy"]]
    val_miou_array = [round(num, 3) for num in state["val_miou"]]

    checkpoint_data = {
        "epoch": state["epoch"],
        "learning_rates": state["learning_rates"][-1],
        "exits": []
    }

    for i in range(exits_num):
        exit_data = {
            "exit_number": exits_num - i,
            "training_loss": train_loss_array[i],
            "training_accuracy": train_acc_array[i],
            "training_miou": train_miou_array[i],
            "validation_loss": val_loss_array[i],
            "validation_accuracy": val_acc_array[i],
            "validation_miou": val_miou_array[i]
        }
        checkpoint_data["exits"].append(exit_data)

    if save_model:
        print("Saving model weights")
        torch.save(state, filename)
    else:
        print("Not saving model weights")

    # with open(logname, 'w') as file:
    #     json.dump(checkpoint_data, file, indent=4)

    # Check if log file already exists
    if os.path.exists(logname):
        # If log file exists, load existing data and append new epoch data
        with open(logname, 'r') as file:
            existing_data = json.load(file)
        existing_data.append(checkpoint_data)
    else:
        # If log file doesn't exist, create a new list with the epoch data
        existing_data = [checkpoint_data]

    # Write the updated data (with appended epoch) back to the JSON log file
    with open(logname, 'w') as file:
        json.dump(existing_data, file, indent=4)





def save_checkpoint_only_split_json(state, filepath, save_model, exits_num, model_num=0):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    cwd = os.getcwd()
    print("Saving checkpoint")

    if not os.path.exists(filepath):
        print('Creating directory', filepath)
        try:
            os.makedirs(filepath)
        except Exception as e:
            print(f"Error creating directory: {e}")

    filename = os.path.join(cwd, f"{filepath}/model{str(model_num)}.pth.tar")
    # logname = os.path.join(cwd, f"{filepath}/log{str(model_num)}.json")
    #
    # train_loss_array = [round(num, 3) for num in state["train_loss"]]
    # val_loss_array = [round(num, 3) for num in state["val_loss"]]
    #
    # checkpoint_data = {
    #     "epoch": state["epoch"],
    #     "learning_rates": state["learning_rates"][-1],
    #     "exits": []
    # }
    #
    # for i in range(exits_num):
    #     exit_data = {
    #         "exit_number": exits_num - i,
    #         "training_loss": train_loss_array[i],
    #         "validation_loss": val_loss_array[i],
    #
    #     }
    #     checkpoint_data["exits"].append(exit_data)

    if save_model:
        print("Saving model weights")
        torch.save(state, filename)
    else:
        print("Not saving model weights")

    # with open(logname, 'w') as file:
    #     json.dump(checkpoint_data, file, indent=4)

    # # Check if log file already exists
    # if os.path.exists(logname):
    #     # If log file exists, load existing data and append new epoch data
    #     with open(logname, 'r') as file:
    #         existing_data = json.load(file)
    #     existing_data.append(checkpoint_data)
    # else:
    #     # If log file doesn't exist, create a new list with the epoch data
    #     existing_data = [checkpoint_data]

    # # Write the updated data (with appended epoch) back to the JSON log file
    # with open(logname, 'w') as file:
    #     json.dump(existing_data, file, indent=4)


def generate_confidence_map(conf_map, prediction_folder):
    color_map = {
        0.0: [255, 255, 0],  # bright yellow
        0.1: [255, 230, 0],  # yellow-orange
        0.2: [255, 204, 0],  # orange
        0.3: [255, 178, 0],  # orange-red
        0.4: [255, 153, 0],  # red-orange
        0.5: [255, 128, 0],  # bright red
        0.6: [230, 0, 115],  # magenta
        0.7: [204, 0, 255],  # purple
        0.8: [150, 0, 255],  # indigo
        0.9: [77, 0, 255],  # dark blue
        1.0: [0, 0, 51]  # dark purple
    }
    try:
        conf_map = np.squeeze(conf_map)
    except RuntimeError as e:
        print(f"Unable to squeeze tensor: {e}")
    output_shape = conf_map.shape[0], conf_map.shape[1], 3
    image = np.zeros(output_shape)
    for x in range(0, output_shape[0]):
        for y in range(0, output_shape[1]):
            conf_value = conf_map[x, y]
            conf_value = conf_value.item()
            rgb = color_map.get(round(conf_value, 1))
            image[x, y] = rgb

    save_numpy_as_image(image, prediction_folder)






def check_acc_one_pic(predictions, y):
    num_correct = 0
    num_pixels = 0
    # outputting a prediction for each individual pixel
    with torch.no_grad():
        # change dim to 1 when batch present
        predictions_acc = torch.argmax(predictions, dim=0)
        y_acc = torch.argmax(y, dim=3)
      #  miou = calculate_mIoU(predictions[0], y[0])
      #   print("size of y:", y_acc.shape)
      #   print("size of predictions:", predictions_acc.shape)
        num_correct += (predictions_acc == y_acc).sum()
        num_pixels += torch.numel(predictions_acc)
        accuracy = num_correct/num_pixels*100
    print("Accuracy of prediction: " + str(round(accuracy.item(), 2))+"%")
    # print("mIoU of prediction: " + str(round(miou.item()*100, 2)) + "%")
    return accuracy


def check_miou_one_pic(predictions, y, num_classes):
    with torch.no_grad():
        predictions = torch.argmax(predictions, dim=0)
        y = torch.argmax(y, dim=2)
        miou = metrics.numpy_iou(predictions, y, num_classes)
    print("mIoU of prediction gpt: " + str(round(miou.item()*100, 2))+"%")
    return miou




def seg_map_to_image(segmap, color_map):
    segmap = np.squeeze(segmap)
    output_shape = segmap.shape[0], segmap.shape[1], 3
    image = np.zeros(output_shape)
    for x in range(0, output_shape[0]):
        for y in range(0, output_shape[1]):
            class_num = segmap[x, y]
            class_num = class_num.item()
            rgb = color_map.get(class_num)
            image[x, y] = rgb
    return image



def save_numpy_as_image(segmap, path):
    image = Image.fromarray(segmap.astype(np.uint8))
    image.save(path)



def draw_losses_graph_joint(train_acc_list, val_acc_list, file, metric_type, exits_num=3):
    scale = 0
    if exits_num == 3:
        train_acc_list = convert_lists(train_acc_list)
        val_acc_list = convert_lists(val_acc_list)
        colors = [['m', '--g'], ['r', '--b'], ['k', '--y']]
        labels = ['Validation_final', 'Training_final', 'Validation_second', 'Training_second',
                    'Validation_first', 'Training_first']
    elif exits_num == 1:
        colors = [['m', '--g']]
        labels = ['Validation', 'Training']
    plt.figure(dpi=1200)
    for i in range(exits_num):
        plt.plot(range(scale, len(val_acc_list[i]) + scale), val_acc_list[i], colors[i][0], label=labels[i * 2])
        plt.plot(range(scale, len(train_acc_list[i]) + scale), train_acc_list[i], colors[i][1], label=labels[i * 2 + 1])
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel(metric_type)
    plt.title(f'{metric_type} w.r.t. num of epochs')
    plt.xticks(range(scale, len(train_acc_list[0]) + scale, floor(len(train_acc_list[0]) / 10)))
    plt.savefig(f'{file}.png')
    plt.close()
    # plt.show()  # display the figure on the screen
    # plt.close()  # close the figure window



def draw_losses_graph_backbone(train_acc_list, val_acc_list, file, metric_type):
    scale = 0
    colors = ['m', '--g']
    plt.legend(['Validation', 'Training'])
    plt.figure(dpi=1200)
    # to lower dimension
    val_acc_list = [i[0] for i in val_acc_list]
    train_acc_list = [i[0] for i in train_acc_list]
    plt.plot(range(scale, len(val_acc_list) + scale), val_acc_list, colors[0], label='Validation')
    plt.plot(range(scale, len(train_acc_list) + scale), train_acc_list, colors[1], label='Training')
    if metric_type.lower() == 'loss':
        plt.legend(loc='upper right')
    else:
        plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel(metric_type)
    plt.title(f'{metric_type} w.r.t. num of epochs')
    plt.xticks(range(scale, len(train_acc_list) + scale, 20))
    plt.savefig(f'{file}.png')
    plt.close()






def draw_costs_graph(cost_list, file):
    scale = 1
    cost_list = convert_lists(cost_list)
    colors = ['m', 'r', 'k']
    plt.figure(dpi=1200)
    for i in range(3):
        plt.plot(range(scale, len(cost_list[i]) + scale), cost_list[i], colors[i])
    plt.legend(['Cost_first', 'Cost_second', 'Cost_final'])
    plt.xlabel('Epoch')
    plt.ylabel('Costs')
    plt.title(f'Exit costs w.r.t. num of epochs')
    plt.xticks(range(scale, len(cost_list[0]) + scale, 20))
    plt.savefig(f'{file}.png')
    plt.close()


def draw_learning_rate(lr_list, file):
    plt.figure(dpi=1200)
    scale = 0
    plt.plot(lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.title(f'LR w.r.t. num of epochs')
    plt.xticks(range(scale, len(lr_list) + scale, 20))
    plt.savefig(f'{file}.png')
    plt.close()
    # plt.show()  # display the figure on the screen
    # plt.close()  # close the figure window


def generate_split_graphs(cloud_flops, mobile_flops, datasize, network_rate, file):

    end_to_end_latency = []

    # Nvidia GeForce 40 series (in Gflops pre second)
    avg_cloud_computational_power = 1_3200_000

    # iPhone13 (in Gflops pre second)
    avg_mobile_computational_power = 1200


    for i in range(0, len(cloud_flops)):


        t_mobile = mobile_flops[i] / avg_mobile_computational_power
        t_uplink = datasize[i][0] / network_rate.value
        t_downlink = datasize[i][1] / network_rate.value
        t_cloud = cloud_flops[i] / avg_cloud_computational_power

        t_total = t_uplink + t_cloud + t_mobile + t_downlink
        end_to_end_latency.append(t_total)

    # Data for different splits and their total latency
    splits = ['On Cloud', 'Encoder1', 'Encoder2', 'Encoder3', 'Encoder4', 'Decoder1', 'Decoder2', 'Decoder3', 'On Mobile']

    # Create bar plot
    plt.figure(dpi=1200)
    plt.bar(splits, end_to_end_latency, color='skyblue')

    # Add labels and title
    plt.xticks(rotation=45)
    plt.xlabel('Split after layer')
    plt.ylabel('Total Latency (s)')
    # plt.title(f'Total Latency of Different Splits for {network_rate.name}')

    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{file}.png')
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
    plt.figure(dpi=1200)
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
    plt.savefig(f'{file}.png')
    plt.close()

def generate_data_size_graph(datasize, file):
    # Data for different splits and their data size
    splits = ['On Cloud', 'Encoder1', 'Encoder2', 'Encoder3', 'Encoder4', 'Decoder1', 'Decoder2', 'Decoder3', 'On Mobile']

    # Create bar plot for data sizes
    if len(datasize[0]) == 2:
        uplink = [i[0] + i[1] for i in datasize]
    elif len(datasize[0]) == 1:
        uplink = [i[0] for i in datasize]
    plt.figure(dpi=1200)
    # print(uplink)
    plt.bar(splits, uplink, color='lightgreen')

    # Add labels and title
    plt.xticks(rotation=45)
    plt.xlabel('Split Location')
    plt.ylabel('Data Size (MB)')
    plt.title('Data Size of Different Splits')

    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{file}.png')
    plt.close()

def convert_lists(list):
    new_list = [[i[0] for i in list], [i[1] for i in list], [i[2] for i in list]]
    return new_list


def get_random_list():
    return [random.randint(5, 50), random.randint(5, 50), random.randint(5, 50)]


def find_threshold(metric_list, exit_percentages, thresholds, file, metric):
    # exited_percentage = exit_counts[exit_position]/sum(exit_counts)*100
    colors = ['m', 'r', 'k']
    fig, ax1 = plt.subplots(dpi=1200)
    ax1.plot(exit_percentages, metric_list, colors[0])
    # plt.legend(['Cost_first', 'Cost_second', 'Cost_final'])
    ax1.set_xlabel('Exited early (%)')
    ax1.set_ylabel(metric)
    ax1.set_title(f'Threshold graph')
    ax1.set_xticks(range(0, 101, 20))
    ax2 = ax1.twinx()
    ax2.plot(exit_percentages, thresholds, colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    ax2.set_ylabel('Thresholds')
    plt.savefig(f'{file}.png')


def check_correlation(metrics, confs, file, metric):
    colors = ['m', 'r', 'k']
    plt.figure(dpi=1200)
    plt.scatter(metrics, confs, c=colors[1])
    plt.xlabel(metric)
    plt.ylabel('Confidence')
    plt.title(f'Correlation')
    plt.savefig(f'{file}.png')


def store_times(times, folder):
    with open(f'{folder}/inference_times.txt', 'w') as file:
        for t in times:
            file.write(str(t) + 'seconds \n')


def create_device(device_from_main):
    global device
    device = device_from_main


def get_device():
    return device


def print_current_dir_and_parent():
    # List directories in the current location
    current_dir = os.getcwd()
    current_files = os.listdir(current_dir)
    print("Files in current directory:", current_files)

    # List directories in the parent directory
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    parent_files = os.listdir(parent_dir)
    print("Files in parent directory:", parent_files)

def calculate_output_padding(input_size, output_size, stride, kernel_size):
    # Calculate the expected output size without output_padding
    expected_output_size_h = (input_size[2] - 1) * stride + kernel_size
    expected_output_size_w = (input_size[3] - 1) * stride + kernel_size

    # Calculate the needed output_padding to match the desired output size
    output_padding_h = max(0, output_size[2] - expected_output_size_h)
    output_padding_w = max(0, output_size[3] - expected_output_size_w)

    return (output_padding_h, output_padding_w)


def get_logs(filepath):
    logs = []
    files = listdir_no_hidden(filepath)

    for filename in files:
        # print(filename)
        with open(os.path.join(filepath, filename), 'r') as file:
            for line in file:
                parsed_line = line.split()
                # Mbps
                throughput = round(int(parsed_line[4]) / int(parsed_line[5]) / 1_000 * 8, 2)
                if throughput != 0:
                    logs.append(throughput)

    return logs



def sum_values_in_dict(dictionary, target_key):
    total_sum_before = 0
    total_sum_after = 0
    found_target_key = False

    pattern1 = r'\bstage\d$'
    pattern2 = r'\bdecoder\d$'


    split_encoder = target_key + ".channel_reduction"
    split_decoder = target_key + ".channel_restoration"

    for key, value in dictionary.items():
        if "split" in key:
            print(key)
        if key == split_decoder:
            # print("Found the target")
            # print(f"Split decoder: | {key} | {value} |")
            total_sum_after += value
            found_target_key = True
        if key == split_encoder:
            # print(f"Split encoder: | {key} | {value} |")
            total_sum_before += value
        matches1 = re.findall(pattern1, key)
        matches2 = re.findall(pattern2, key)
        if matches1 or matches2:
            if target_key == "mobile":
                total_sum_before += value
            elif target_key == "server":
                total_sum_after += value
            elif not found_target_key:
                # print(f"Before: | {key} | {value} |")
                total_sum_before += value
            elif found_target_key:
                # print(f"After: | {key} | {value} |")
                total_sum_after += value

    if not found_target_key and target_key != "server" and target_key != "mobile":
        raise ValueError(f"Wanted split layer is not present in the split {target_key}")
    return total_sum_before, total_sum_after


def sum_all_values_in_dict(dictionary):
    total_sum = 0

    for key, value in dictionary.items():
        total_sum += value

    return total_sum


def calculate_before_and_after_flops(dictionary, split_points):
    mobile_flops, server_flops = 0, 0
    spilt_names = ["split1", "split2", "split3", "split4"
                   , "split5", "split6", "split7"]
    if split_points[0]:
        # whole network on server - no splits
        _, server_flops = sum_values_in_dict(dictionary, "server")
    else:
        split_location = find_index_of_split(split_points)
        print(f"Split location: {split_location}")
        if split_location == -1:
            # whole network on mobile - no splits
            mobile_flops, _ = sum_values_in_dict(dictionary, "mobile")
        else:
            target = spilt_names[split_location - 1]
            mobile_flops, server_flops = sum_values_in_dict(dictionary, target)
    return mobile_flops, server_flops


def set_next_spilt(array):
    new_array = copy.deepcopy(array)
    for i in range(len(array)):
        if new_array[i]:
            new_array[i] = False
            if i < len(new_array) - 1:
                new_array[i + 1] = True
                return new_array

# finding first True in array
def find_index_of_split(split_points):
    for i in range(0, len(split_points)):
        if split_points[i]:
            return i
    return -1



def create_all_split_possibilities(send_result_back):
    if send_result_back:
        all_split_possibilties = [[True, False, False, False, False, False, False, False, True]]
    else:
        # last split always False -> is only used to send results back
        all_split_possibilties = [[True, False, False, False, False, False, False, False, False]]

    for i in range(0, 7):
        next_split_array = set_next_spilt(all_split_possibilties[i])
        all_split_possibilties.append(next_split_array)
    all_split_possibilties.append([False, False, False, False, False, False, False, False, False])
    return all_split_possibilties


