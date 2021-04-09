import math
import time
from random import choice

import cv2
import h5py
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.optim import Optimizer, Adam, SGD
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm.auto import tqdm as tq

from craterdetection.detection.loss_functions.dice import dice_coefficient
from craterdetection.detection.pre_processing import calculate_cdf, match_histograms


class CraterDataset(Dataset):
    transform = {
        'images': transforms.Compose([
            transforms.ToTensor()
        ]),
        'masks': transforms.Compose([
            transforms.ToTensor()
        ])
    }

    def __init__(self,
                 file_path,
                 group,
                 device=None,
                 histogram_matching=True
                 ):
        self.file_path = file_path
        self.group = group
        self.dataset = None
        self.device = device

        if histogram_matching:
            with h5py.File(self.file_path, 'r') as hf:
                images = hf['training/images'][:]
                images = (images / np.max(images, axis=(-2, -1))[..., None, None])*255
                ref_hist, _ = np.histogram(images.flatten(), 256, [0, 256])
                self.ref_cdf = calculate_cdf(ref_hist)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')

        images = self.dataset[self.group]["images"][idx]
        masks = self.dataset[self.group]["masks"][idx]

        images = (images / np.max(images, axis=(-2, -1))[..., None, None])*255

        # clahe = cv2.createCLAHE(tileGridSize=(8, 8))

        for i in range(len(images)):
            if self.ref_cdf is not None:
                images[i] = match_histograms(images[i], ref_cdf=self.ref_cdf)

            images[i] = cv2.GaussianBlur(images[i], (3, 3), 1)
            # images[i] = clahe.apply(images[i].astype(np.uint8))

        images = torch.as_tensor(images)
        masks = torch.as_tensor(masks, dtype=torch.float32)

        if self.device is not None:
            return images.to(self.device), masks.to(self.device)
        else:
            return images, masks

    def random(self):
        return self.__getitem__(
            np.random.randint(0, len(self))
        )

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            return len(f[self.group]['images'])

    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()


# https://www.kaggle.com/dhananjay3/image-segmentation-from-scratch-in-pytorch
class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(tensor1=grad, tensor2=grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


def get_trial(model, lr_list, momentum_list, loss_function_list, optimizer_list):
    lr = choice(lr_list)
    momentum = choice(momentum_list)

    loss_function = choice(loss_function_list)
    lf_params = {}
    # if loss_function == BCEDiceLoss:
    #     lambda_dice = choice(lambda_dice_list)
    #     lambda_bce = choice(lambda_bce_list)
    #     eps = choice(eps_list)
    #     lf_params = dict(lambda_dice=lambda_dice, lambda_bce=lambda_bce, eps=eps)
    loss_function = loss_function(**lf_params)

    optimizer = choice(optimizer_list)
    opt_params = dict(lr=lr)
    if optimizer == SGD:
        opt_params['momentum'] = momentum
    optimizer = optimizer(model.parameters(), **opt_params)

    return loss_function, lf_params, optimizer, opt_params


def hypersearch(model_callable, num_epochs, num_trials, train_loader, validation_loader):
    with mlflow.start_run(run_name="Hyperparameter Search"):
        for _ in range(num_trials):
            with mlflow.start_run(nested=True):
                model = model_callable()
                model.cuda()

                loss_function, lf_params, optimizer, opt_params = get_trial(model)

                mlflow.log_param('optimizer', type(optimizer).__name__)
                mlflow.log_param('loss_function', type(loss_function).__name__)
                for k, v in opt_params.items():
                    mlflow.log_param(k, v)

                train_loss_list = []
                valid_loss_list = []
                dice_score_list = []

                for e in range(1, num_epochs + 1):
                    print(f'\n-----Epoch {e} started-----\n')

                    since = time.time()

                    train_loss, valid_loss, dice_score = 0, 0, 0

                    model.train()

                    bar = tq(train_loader, desc=f"Training [{e}]", postfix={"train_loss": 0.0})
                    for batch, (images, masks) in enumerate(bar, 1):
                        optimizer.zero_grad()

                        with autocast():
                            pred = model(images)
                            loss = loss_function(pred, masks)

                        loss.backward()
                        train_loss += loss.item() * images.size(0)

                        optimizer.step()
                        bar.set_postfix(ordered_dict={"train_loss": loss.item()})

                    model.eval()
                    del images, masks
                    with torch.no_grad():
                        bar = tq(validation_loader, desc=f"Validation [{e}]",
                                 postfix={"valid_loss": 0.0, "dice_score": 0.0})
                        for images, masks in bar:
                            with autocast():
                                pred = model(images)
                                loss = loss_function(pred, masks)

                            valid_loss += loss.item() * images.size(0)
                            dice_cof = dice_coefficient(pred, masks).item()
                            dice_score += dice_cof * images.size(0)
                            bar.set_postfix(ordered_dict={"valid_loss": loss.item(), "dice_score": dice_cof})
                    # calculate average losses
                    train_loss = train_loss / len(train_loader.dataset)
                    valid_loss = valid_loss / len(validation_loader.dataset)
                    dice_score = dice_score / len(validation_loader.dataset)
                    train_loss_list.append(train_loss)
                    valid_loss_list.append(valid_loss)
                    dice_score_list.append(dice_score)

                    mlflow.log_metric("train_loss", train_loss, step=e)
                    mlflow.log_metric("valid_loss", valid_loss, step=e)
                    mlflow.log_metric("dice_score", dice_score, step=e)

                    time_elapsed = time.time() - since
                    print(
                        f"\nSummary:\n",
                        f"\tEpoch: {e}/{num_epochs}\n",
                        f"\tAverage train loss: {train_loss}\n",
                        f"\tAverage validation loss: {valid_loss}\n",
                        f"\tAverage Dice score: {dice_score}\n",
                        f"\tDuration: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
                    )
                    print(f'-----Epoch {e} finished.-----\n')
