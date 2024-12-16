import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import utils.utils as utils
import TransformerTrainingStrategy



def train(loader, model, optimizer, loss_func):

    loop = tqdm(loader, leave=False)
    counter = 0
    total_acc = 0
    total_miou = 0

    for batch_idx, (data, targets) in enumerate(loop):
        loss = None
        # convert to float for entropy
        data = data.to(device=utils.get_device())
        targets = targets.float()
        # shifting channels
        targets = targets.to(device=utils.get_device())
        targets_permuted = targets
        if not isinstance(model.strategy, TransformerTrainingStrategy.TrainSplitOnly):
            targets_permuted = targets.permute(0, 3, 1, 2)
        # forward
        predictions = model(data)
        # loss = loss_func(predictions, targets)
        loss = model.calculate_loss(predictions, targets_permuted, loss_func)
        acc, miou = model.check_batch_acc_miou_loss_no_loader(predictions, targets)
        total_acc += acc.item()
        total_miou += miou.item()

        optimizer.zero_grad()
        if isinstance(loss, list):
            # if loss is a list, iterate through each element
            for idx, i in enumerate(loss):
                if idx == len(loss) - 1:
                    i.backward()
                else:
                    # retain computation graph except for last call to backward
                    i.backward(retain_graph=True)
            loop.set_postfix(loss=sum(loss).item())
        else:
            # if loss is a single value, directly apply backward() to it
            loss.backward()
            loop.set_postfix(loss=loss.item())
        # weights update
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        data = data.to("cpu")
        targets = targets.to("cpu")
        counter += 1
    # print('Current learning rate', optimizer.param_groups[0]["lr"])
    avg_batch_acc = (total_acc / counter)
    avg_batch_miou = (total_miou / counter) * 100
    torch.cuda.empty_cache()
    return [avg_batch_acc], [avg_batch_miou], [loss.item()]

