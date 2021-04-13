import math
from random import choice

import cv2
import h5py
import numpy as np
import torch
from torch.optim import Optimizer, SGD
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from typing import Tuple

from craterdetection.detection.pre_processing import calculate_cdf, match_histograms


class CraterDataset(Dataset):
    def __init__(self,
                 file_path,
                 group,
                 stretch=1,
                 histogram_matching=False,
                 gaussian_blur=False,
                 clahe=False
                 ):
        self.file_path = file_path
        self.group = group
        self.stretch = stretch
        self.dataset = None
        self.gaussian_blur = gaussian_blur
        self.clahe = clahe

        if histogram_matching:
            with h5py.File(self.file_path, 'r') as hf:
                images = hf['training/images'][:]
                images = (images / np.max(images, axis=(-2, -1))[..., None, None]) * self.stretch
                ref_hist, _ = np.histogram(images.flatten(), 256, [0, 256])
                self.ref_cdf = calculate_cdf(ref_hist)
        else:
            self.ref_cdf = None

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')

        images = self.dataset[self.group]["images"][idx]
        masks = self.dataset[self.group]["masks"][idx]

        images = (images / np.max(images, axis=(-2, -1))[..., None, None]) * self.stretch

        if self.gaussian_blur or self.ref_cdf is not None or self.clahe:
            clahe = cv2.createCLAHE(tileGridSize=(8, 8))
            for i in range(len(images)):
                if self.ref_cdf is not None:
                    images[i] = match_histograms(images[i], ref_cdf=self.ref_cdf)

                if self.gaussian_blur:
                    images[i] = cv2.GaussianBlur(images[i], (3, 3), 1)

                if self.clahe:
                    images[i] = clahe.apply(images[i].astype(np.uint8))

        images = torch.as_tensor(images)
        masks = torch.as_tensor(masks, dtype=torch.float32)

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


def collate_fn(batch):
    return tuple(zip(*batch))


class CraterInstanceDataset(CraterDataset):
    def __init__(self, *args, **kwargs):
        super(CraterInstanceDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        images, mask = super(CraterInstanceDataset, self).__getitem__(idx)
        images, mask = images.numpy(), mask.numpy()

        mask: np.ndarray = mask.astype(int)

        obj_ids = np.unique(mask)[1:]
        masks: np.ndarray = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)

        boxes = np.empty((num_objs, 4), int)
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes[i] = np.array([xmin, ymin, xmax, ymax])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area_filter = area > 4

        masks, obj_ids, boxes, area = map(lambda x: x[area_filter], (masks, obj_ids, boxes, area))

        num_objs = len(obj_ids)

        images = torch.as_tensor(images, dtype=torch.float32)

        boxes = torch.as_tensor(boxes, dtype=torch.int64)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        area = torch.as_tensor(area, dtype=torch.float32)
        image_id = torch.tensor([idx])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return images, target

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


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
                exp_avg.mul_(beta1).add_(grad, 1 - beta1)

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


def load_checkpoint(model, path):
    if not torch.cuda.is_available():
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint.pop('model_state_dict')

    return model, checkpoint


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
