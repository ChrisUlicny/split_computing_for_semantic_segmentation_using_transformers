import statistics

import torch
import torch.nn.functional as F
import torch.distributions as dist


def calculate_confidence(prediction, threshold=None, method='avg'):
    # shape : 1 x classes x height x width
    prediction = torch.squeeze(prediction)
    # shape : classes x height x width
    prediction_percentages = F.softmax(input=prediction, dim=0)

    # entropy confidence
    # entropy = -torch.sum(prediction_percentages * torch.log(prediction_percentages), dim=0)
    # max_percentages = 1.0 - entropy

    # max confidence
    max_percentages = torch.max(prediction_percentages, dim=0)
    max_percentages = max_percentages.values

    # margin confidence
    # sorted_probs, _ = torch.sort(prediction_percentages, descending=True, dim=0)
    # max_percentages = sorted_probs[0] - sorted_probs[1]

    confidence_map = max_percentages.detach().cpu().numpy()
    if threshold is None:
        max_percentages_np = max_percentages.detach().cpu().numpy().reshape((-1,))
        confidence = choose_function(max_percentages_np, method)
        return confidence, confidence_map
    else:
        count = 0
        for i in range(0, prediction.shape[1]):
            for j in range(0, prediction.shape[2]):
                if max_percentages[i][j].item() > threshold:
                    count += 1
        num_pixels = prediction.shape[1]*prediction.shape[2]
        print(f"Confidence calculation: {count}/{num_pixels} = {round(count/num_pixels,2)}%")
        return count/num_pixels


def choose_function(data, method):
    if method == 'avg':
        return calculate_confidence_avg(data)
    if method == 'median':
        return calculate_confidence_median(data)
    else:
        raise ValueError('Invalid confidence evaluation value encountered')



def calculate_confidence_median(prediction):
    confidence = statistics.median(prediction)
    # print(f"Median confidence calculation: {round(confidence*100,2)}%")
    return confidence

def calculate_confidence_avg(prediction):
    confidence = statistics.mean(prediction)
    return confidence
