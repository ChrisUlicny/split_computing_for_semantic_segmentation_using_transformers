import torch
import numpy as np
import utils.utils as utils
import torch.nn.functional as F

def check_batch_acc_miou_loss(model, loader, num_classes, loss_func=None):
    num_correct = 0
    num_pixels = 0
    accuracy = 0
    counter = 0
    total_loss = 0
    miou = 0
    miou_list = [0, 0]
    losses_exits = [0, 0]
    accuracies_exits = [0, 0]
    mious_exits = [0, 0]
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            all_predictions = []
            x = x.to(device=utils.get_device())
            y = y.to(device=utils.get_device())
            predictions = model(x)
            loss = model.calculate_loss(predictions, y.permute(0, 3, 1, 2), loss_func)
            if isinstance(loss, list):
                repeat = 2
                if len(predictions) > 2:
                    # distillation
                    predictions = [predictions[0], predictions[1]]
                for idx in range(2):
                    losses_exits[idx] += loss[idx]
                for p in predictions:
                    all_predictions.append(torch.argmax(p, dim=1))
                y = torch.argmax(y, dim=3)
                bs = all_predictions[0].shape[0]
                for idx, p in enumerate(all_predictions):
                    num_correct += (p == y).sum()
                    num_pixels += torch.numel(p)
                    accuracies_exits[idx] += num_correct / num_pixels * 100
                for i in range(0, bs):
                    for idx, p in enumerate(all_predictions):
                        mious_exits[idx] += calculate_mIoU(p[i], y[i], num_classes)
                mious_exits = [i / bs for i in mious_exits]
                miou_list = [miou_list[i] + mious_exits[i] for i in range(0, len(mious_exits))]
            else:
                if isinstance(predictions, tuple):
                    # training splits
                    predictions = predictions[0]
                    y = predictions[1]
                repeat = 1
                miou_batch = 0
                total_loss = total_loss + loss
                predictions = torch.argmax(predictions, dim=1)
                y = torch.argmax(y, dim=3)
                num_correct += (predictions == y).sum()
                num_pixels += torch.numel(predictions)
                accuracy = accuracy + num_correct / num_pixels * 100
                bs = predictions.shape[0]
                for i in range(0, bs):
                    miou_batch += calculate_mIoU(predictions[i], y[i], num_classes)

                miou_batch = miou_batch / bs
                miou += miou_batch

            counter += 1
            x = x.to("cpu")
            y = y.to("cpu")

    model.train()
    if repeat == 1:
        avg_batch_acc = (accuracy / counter).item()
        # print("Average accuracy across all batches: " + str(round(avg_batch_acc, 2)) + "%")

        avg_batch_miou = (miou / counter).item() * 100
        # print("Average mIoU across all batches: " + str(round(avg_batch_miou, 2)) + "%")

        avg_loss = (total_loss / counter).item()
        # print("Average loss across all batches: " + str(round(avg_loss, 3)))
        # print(avg_batch_acc, avg_batch_miou, avg_loss)
        return [avg_batch_acc], [avg_batch_miou], [avg_loss]
    elif repeat == 2:
        avg_batch_accs = []
        avg_losses = []
        avg_batch_mious = []

        for idx, p in enumerate(all_predictions):

            avg_batch_mious.append((miou_list[idx] / counter).item() * 100)
            # print("Average mIoU across all batches exit " + str(idx + 1) + " : " + str(
            #     round(avg_batch_mious[idx], 2)) + "%")

            avg_batch_accs.append((accuracies_exits[idx] / counter).item())
            # print("Average accuracy across all batches exit " + str(idx + 1) + " : " + str(
            #     round(avg_batch_accs[idx], 2)) + "%")

            if loss_func is not None:
                avg_losses.append((losses_exits[idx] / counter).item())
                # print("Average loss across all batches exit " + str(idx + 1) + " : " + str(
                #     round(avg_losses[idx], 3)))
        return avg_batch_accs, avg_batch_mious, avg_losses



def check_batch_acc_miou_loss_joint(model, loader, num_classes, loss_func=None):
    num_correct = 0
    num_pixels = 0
    counter = 0
    losses_exits = [0, 0, 0]
    accuracies_exits = [0, 0, 0]
    mious_exits = [0, 0, 0]
    miou = [0, 0, 0]
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            all_predictions = []
            x = x.to(device=utils.get_device())
            y = y.to(device=utils.get_device())
            predictions = model(x)
            for p in predictions:
                all_predictions.append(torch.argmax(p[0], dim=1))

            if loss_func is not None:
                for idx in range(len(predictions)):
                    losses_exits[idx] += model.calculate_loss(predictions, y.permute(0, 3, 1, 2), loss_func, separate=True)[idx]

            y = torch.argmax(y, dim=3)
            bs = all_predictions[0].shape[0]
            # accuracy_exits = 0
            for idx, p in enumerate(all_predictions):
                num_correct += (p == y).sum()
                num_pixels += torch.numel(p)
                accuracies_exits[idx] += num_correct/num_pixels*100
                # accuracy_exits = accuracy_exits + (num_correct/num_pixels*100)
            # accuracy += accuracy_exits / len(all_predictions)
            for i in range(0, bs):
                if len(predictions) > 1:
                    for idx, p in enumerate(all_predictions):
                        mious_exits[idx] += calculate_mIoU(p[i], y[i], num_classes)
            mious_exits = [i/bs for i in mious_exits]
            miou = [miou[i] + mious_exits[i] for i in range(0, len(mious_exits))]
            counter += 1
            x = x.to("cpu")
            y = y.to("cpu")
    model.train()
    avg_batch_accs = []
    avg_losses = []
    avg_batch_mious = []

    for idx, p in enumerate(all_predictions):

        avg_batch_mious.append((miou[idx] / counter).item() * 100)
        # print("Average mIoU across all batches exit " + str(len(predictions) - idx) + " : " + str(
        #     round(avg_batch_mious[idx], 2)) + "%")

        avg_batch_accs.append((accuracies_exits[idx]/counter).item())
        # print("Average accuracy across all batches exit " + str(len(predictions)-idx) + " : " + str(round(avg_batch_accs[idx], 2)) + "%")

        if loss_func is not None:
            avg_losses.append((losses_exits[idx] / counter).item())
            # print("Average loss across all batches exit " + str(len(predictions)-idx) + " : " + str(round(avg_losses[idx], 3)))
    if loss_func is not None:
        return avg_batch_accs, avg_batch_mious, avg_losses
    return avg_batch_accs, avg_batch_mious


def check_batch_miou(model, loader, num_classes):

    counter = 0
    miou = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            miou_batch = 0
            x = x.to(device=utils.get_device())
            y = y.to(device=utils.get_device())
            # for the knowledge distillation case
            try:
                predictions, _ = model(x)
            except ValueError:
                predictions = model(x)
            predictions = torch.argmax(predictions, dim=1)
            bs = predictions.shape[0]
            y = torch.argmax(y, dim=3)
            for i in range(0, bs):
                miou_batch += calculate_mIoU(predictions[i], y[i], num_classes)

            miou_batch = miou_batch / bs
            miou += miou_batch
            counter += 1
            x = x.to("cpu")
            y = y.to("cpu")

    model.train()
    avg_batch_miou = (miou / counter).item() * 100
    print("Average mIoU across all batches: " + str(round(avg_batch_miou, 2)) + "%")
    return avg_batch_miou


def check_batch_miou_joint(model, loader, num_classes):
    counter = 0
    mious_exits = [0, 0, 0]
    miou = [0, 0, 0]
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            all_predictions = []
            x = x.to(device=utils.get_device())
            y = y.to(device=utils.get_device())
            predictions = model(x)
            for p in predictions:
                all_predictions.append(torch.argmax(p[0], dim=1))
            bs = all_predictions[0].shape[0]
            y = torch.argmax(y, dim=3)
            for i in range(0, bs):
                if len(predictions) > 1:
                    for idx, p in enumerate(all_predictions):
                        mious_exits[idx] += calculate_mIoU(p[i], y[i], num_classes)
            mious_exits = [i/bs for i in mious_exits]
            miou = [miou[i] + mious_exits[i] for i in range(0, len(mious_exits))]
            counter += 1
            x = x.to("cpu")
            y = y.to("cpu")

    model.train()
    avg_batch_mious = []
    for idx, p in enumerate(all_predictions):
        avg_batch_mious.append((miou[idx] / counter).item() * 100)
        print("Average mIoU across all batches exit " + str(len(predictions) - idx) + " : " + str(
            round(avg_batch_mious[idx], 2)) + "%")

    return avg_batch_mious



def calculate_mIoU(prediction, y, num_classes):
    pred = prediction.cpu().numpy()
    y_np = y.cpu().numpy()
    pred_1d = pred.reshape((prediction.shape[0] * prediction.shape[1], ))
    y_1d = y_np.reshape((y.shape[0] * y.shape[1],))
    pred_frequency_map = np.bincount(pred_1d, minlength=num_classes)
    actual_frequency_map = np.bincount(y_1d, minlength=num_classes)
    # category of every pixel
    categories = num_classes * y_1d + pred_1d
    # num of pixels belonging to a particular category
    confusion_matrix = np.bincount(categories, minlength=num_classes * num_classes)
    confusion_matrix = confusion_matrix.reshape((num_classes, num_classes))
    intersection = np.diag(confusion_matrix)
    union = pred_frequency_map + actual_frequency_map - intersection
    iou = intersection / union
    # iou = intersection / union
    return np.nanmean(iou)




def check_batch_acc_miou_loss_no_loader(model, x, y, num_classes, loss_func=None):
    num_correct = 0
    num_pixels = 0
    accuracy = 0
    counter = 0
    total_loss = 0
    miou = 0
    miou_list = [0, 0]
    losses_exits = [0, 0]
    accuracies_exits = [0, 0]
    mious_exits = [0, 0]
    model.eval()

    with torch.no_grad():
        all_predictions = []
        x = x.to(device=utils.get_device())
        y = y.to(device=utils.get_device())
        predictions = model(x)
        print(predictions)
        loss = model.calculate_loss(predictions, y.permute(0, 3, 1, 2), loss_func)
        if isinstance(loss, list):
            repeat = 2
            if len(predictions) > 2:
                # distillation
                predictions = [predictions[0], predictions[1]]
            for idx in range(2):
                losses_exits[idx] += loss[idx]
            for p in predictions:
                all_predictions.append(torch.argmax(p, dim=1))
            y = torch.argmax(y, dim=3)
            bs = all_predictions[0].shape[0]
            for idx, p in enumerate(all_predictions):
                num_correct += (p == y).sum()
                num_pixels += torch.numel(p)
                accuracies_exits[idx] += num_correct / num_pixels * 100
            for i in range(0, bs):
                for idx, p in enumerate(all_predictions):
                    mious_exits[idx] += calculate_mIoU(p[i], y[i], num_classes)
            mious_exits = [i / bs for i in mious_exits]
            miou_list = [miou_list[i] + mious_exits[i] for i in range(0, len(mious_exits))]
        else:
            if isinstance(predictions, tuple):
                # training splits
                predictions = predictions[0]
                y = predictions[1]
            repeat = 1
            miou_batch = 0
            total_loss = total_loss + loss
            predictions = torch.argmax(predictions, dim=1)
            y = torch.argmax(y, dim=3)
            num_correct += (predictions == y).sum()
            num_pixels += torch.numel(predictions)
            accuracy = accuracy + num_correct / num_pixels * 100
            bs = predictions.shape[0]
            for i in range(0, bs):
                miou_batch += calculate_mIoU(predictions[i], y[i], num_classes)

            miou_batch = miou_batch / bs
            miou += miou_batch

        counter += 1
        x = x.to("cpu")
        y = y.to("cpu")

    model.train()
    if repeat == 1:
        avg_batch_acc = (accuracy / counter).item()
        # print("Average accuracy across all batches: " + str(round(avg_batch_acc, 2)) + "%")

        avg_batch_miou = (miou / counter).item() * 100
        # print("Average mIoU across all batches: " + str(round(avg_batch_miou, 2)) + "%")

        avg_loss = (total_loss / counter).item()
        # print("Average loss across all batches: " + str(round(avg_loss, 3)))
        print(avg_batch_acc, avg_batch_miou)
        return [avg_batch_acc], [avg_batch_miou], [avg_loss]
    elif repeat == 2:
        avg_batch_accs = []
        avg_losses = []
        avg_batch_mious = []

        for idx, p in enumerate(all_predictions):

            avg_batch_mious.append((miou_list[idx] / counter).item() * 100)
            # print("Average mIoU across all batches exit " + str(idx + 1) + " : " + str(
            #     round(avg_batch_mious[idx], 2)) + "%")

            avg_batch_accs.append((accuracies_exits[idx] / counter).item())
            # print("Average accuracy across all batches exit " + str(idx + 1) + " : " + str(
            #     round(avg_batch_accs[idx], 2)) + "%")

            if loss_func is not None:
                avg_losses.append((losses_exits[idx] / counter).item())
                # print("Average loss across all batches exit " + str(idx + 1) + " : " + str(
                #     round(avg_losses[idx], 3)))
        return avg_batch_accs, avg_batch_mious, avg_losses
