import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import MSELoss
import utils
import math


def loss_func_joint_training(output, targets, separate=False):
    losses, costs = [], []
    loss_func = nn.CrossEntropyLoss()
    for idx, o in enumerate(output):
        prediction, cost = o[0], o[1]
        loss = cost * loss_func(prediction, targets)
        costs.append(cost)
        losses.append(loss)
    if separate:
        return losses
    else:
        # to compensate for magnitude
        return sum(losses)/sum(costs)


def loss_func_exits(output, targets):
    two_exit_losses = []
    loss_func = nn.CrossEntropyLoss()
    for idx, prediction in enumerate(output):
        loss = loss_func(prediction, targets)
        two_exit_losses.append(loss)
    return two_exit_losses



def dist_exits_loss_func(output, targets, alpha):
    MSEloss = nn.MSELoss(reduction='mean').to(device=utils.get_device())
    loss_func = nn.CrossEntropyLoss()
    first_exit_pred = output[0]
    second_exit_pred = output[1]
    teacher_pred = output[2]
    teacher_pred = teacher_pred.detach()

    # classification loss calculation
    classification_loss_first = loss_func(first_exit_pred, targets)
    classification_loss_second = loss_func(second_exit_pred, targets)

    # distillation loss calculation
    dist_loss_first = MSEloss(first_exit_pred, teacher_pred)
    dist_loss_second = MSEloss(second_exit_pred, teacher_pred)

    # final loss calculation
    final_loss_first = alpha * dist_loss_first + (1 - alpha) * classification_loss_first
    final_loss_second = alpha * dist_loss_second + (1 - alpha) * classification_loss_second

    loss = [final_loss_first, final_loss_second]
    return loss


def loss_func_joint_distillation(output, targets, alpha, separate=False):
    MSEloss = nn.MSELoss(reduction='mean').to(device=utils.get_device())
    classification_loss = 0
    predictions = []
    costs = []
    losses = []
    num_exits = len(output)

    # classification loss calculation
    loss_func = nn.CrossEntropyLoss()
    for o in output:
        prediction, cost = o[0], o[1]
        predictions.append(prediction)
        costs.append(cost)
        # print(prediction.shape)
        loss = loss_func(prediction, targets)
        losses.append(loss)
        loss = cost * loss
        classification_loss += loss

    avg_cost_to_normalize = sum(costs) / len(costs)
    classification_loss /= avg_cost_to_normalize


    target_output = predictions[0].detach()
    dist_loss_second = MSEloss(predictions[1], target_output)
    dist_loss_first = MSEloss(predictions[2], target_output)
    distillation_loss = dist_loss_first + dist_loss_second
    final_loss = (1 - alpha) * classification_loss + (alpha) * distillation_loss
    if separate == True:
        return losses

    return final_loss


def loss_func_exit_ensemble(output, targets, alpha=1, gamma=1, lamda=1, separate=False):
    MSEloss = nn.MSELoss(reduction='mean').to(device=utils.get_device())
    classification_loss = 0
    distillation_loss = 0
    lambdas = []
    losses = []
    predictions = []
    costs = []
    num_exits = len(output)
    for i in range(1, num_exits+1):
        lambdas.append(pow(lamda, num_exits-i))
    # classification loss calculation
    loss_func = nn.CrossEntropyLoss()
    for idx, o in enumerate(output):
        prediction, cost = o[0], o[1]
        predictions.append(prediction)
        costs.append(cost)
        # print(prediction.shape)
        loss = cost * loss_func(prediction, targets)
        losses.append(loss)
        balancing_constant = 1 + alpha - pow(gamma, idx)
        classification_loss += loss * balancing_constant
    # cum_loss = cum_loss/num_exits
    avg_cost_to_normalize = sum(costs) / len(costs)
    classification_loss /= avg_cost_to_normalize

    # ensemble
    target_output = ((lambdas[0] * predictions[0]/3 + lambdas[1] *
                      predictions[1]/3 + lambdas[2] * predictions[2]/3)/sum(lambdas)).detach()

    temp_loss = MSEloss(predictions[0], target_output)
    losses[0] += temp_loss
    distillation_loss += temp_loss * pow(gamma, 0)
    temp_loss = MSEloss(predictions[1], target_output)
    losses[1] += temp_loss
    distillation_loss += temp_loss * pow(gamma, 1)
    temp_loss = MSEloss(predictions[2], target_output)
    losses[2] += temp_loss
    distillation_loss += temp_loss * pow(gamma, 2)
    final_loss = classification_loss + distillation_loss

    if separate == True:
        return losses
    return final_loss


# def loss_func_exit_ensemble_not_weighted(output, targets, alpha=1, gamma=1.15, lamda=1.6, separate=False):
#     MSEloss = nn.MSELoss(reduction='mean').to(device=utils.get_device())
#     classification_loss = 0
#     distillation_loss = 0
#     losses = []
#     predictions = []
#     costs = []
#     # classification loss calculation
#     loss_func = nn.CrossEntropyLoss()
#     for idx, o in enumerate(output):
#         prediction, cost = o[0], o[1]
#         predictions.append(prediction)
#         costs.append(cost)
#         # print(prediction.shape)
#         loss = cost * loss_func(prediction, targets)
#         losses.append(loss)
#         classification_loss += loss
#     # cum_loss = cum_loss/num_exits
#     avg_cost_to_normalize = sum(costs) / len(costs)
#     classification_loss /= avg_cost_to_normalize
#
#
#     # ensemble
#     target_output = (predictions[0]/3 + predictions[1]/3 + predictions[2]/3).detach()
#
#     temp_loss = MSEloss(predictions[0], target_output)
#     losses[0] += temp_loss
#     distillation_loss += temp_loss
#     temp_loss = MSEloss(predictions[1], target_output)
#     losses[1] += temp_loss
#     distillation_loss += temp_loss
#     temp_loss = MSEloss(predictions[2], target_output)
#     losses[2] += temp_loss
#     distillation_loss += temp_loss
#     final_loss = (1-alpha) * classification_loss + alpha * distillation_loss
#
#     if separate == True:
#         return losses
#     return final_loss
