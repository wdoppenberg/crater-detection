import torch
import torch.nn as nn
from torch.optim import SGD

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

import h5py


class CraterDataset(Dataset):
    pass


def train_step(model, images, masks, loss_function, learning_rate=1e-2, momentum=0.9):
    model.train()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer.zero_grad()

    masks_pred = model(images)
    loss = loss_function(masks_pred, masks)
    loss.backward()
    optimizer.step()
    return loss.item()
