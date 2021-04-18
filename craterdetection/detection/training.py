import math
import os
import time
from random import choice
from statistics import mean
from typing import Tuple, Dict, Iterable

import cv2
import h5py
import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.cuda.amp import autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm as tq

from .pre_processing import calculate_cdf, match_histograms
from .visualisation import draw_patches


class CraterDataset(Dataset):
    def __init__(self,
                 file_path,
                 group,
                 transforms=None,
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

    def __getitem__(self, idx: ...) -> Tuple[torch.Tensor, torch.Tensor]:
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


def collate_fn(batch: Iterable):
    return tuple(zip(*batch))


class CraterInstanceDataset(CraterDataset):
    def __init__(self, min_area=4, *args, **kwargs):
        super(CraterInstanceDataset, self).__init__(*args, **kwargs)
        self.min_area = min_area

    def __getitem__(self, idx: ...) -> Tuple[torch.Tensor, Dict]:
        images, mask = super(CraterInstanceDataset, self).__getitem__(idx)
        mask: torch.Tensor = mask.int()

        obj_ids = mask.unique()[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)

        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)

        for i in range(num_objs):
            pos = torch.where(masks[i])
            xmin = pos[1].min()
            xmax = pos[1].max()
            ymin = pos[0].min()
            ymax = pos[0].max()
            boxes[i] = torch.tensor([xmin, ymin, xmax, ymax])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area_filter = area > self.min_area

        masks, obj_ids, boxes, area = map(lambda x: x[area_filter], (masks, obj_ids, boxes, area))

        num_objs = len(obj_ids)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = masks.int()
        image_id = torch.tensor([idx])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = dict(
            boxes=boxes,
            labels=labels,
            masks=masks,
            image_id=image_id,
            area=area,
            iscrowd=iscrowd
        )

        return images, target

    @staticmethod
    def collate_fn(batch: Iterable):
        return collate_fn(batch)


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


def get_dataloaders(dataset_path: str, batch_size: int = 10, num_workers: int = 4) -> \
        Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = CraterInstanceDataset(file_path=dataset_path, group="training")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
                              shuffle=True)

    validation_dataset = CraterInstanceDataset(file_path=dataset_path, group="validation")
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=0,
                                   collate_fn=collate_fn)

    test_dataset = CraterInstanceDataset(file_path=dataset_path, group="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn,
                             shuffle=True)

    return train_loader, validation_loader, test_loader


def train_model(model: nn.Module, num_epochs: int, dataset_path: str, initial_lr=1e-2, run_id: str = None,
                scheduler=None, batch_size: int = 10, momentum: float = 0.5, weight_decay: float = 1e-5,
                num_workers: int = 4, device=None) -> None:
    train_loader, validation_loader, test_loader = get_dataloaders(dataset_path, batch_size, num_workers)

    pretrained = run_id is not None

    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("crater-detection")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained:
        checkpoint = mlflow.pytorch.load_state_dict(f"runs:/{run_id}/checkpoint")

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = SGD(params, lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        checkpoint = dict()
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = SGD(params, lr=initial_lr, momentum=0.5, weight_decay=1e-7)

    if scheduler is None:
        scheduler = StepLR(optimizer, step_size=10)

    tracked_params = ('momentum', 'weight_decay', 'dampening')

    name = "Mask RCNN"
    name += " | Edge" if "edge" in dataset_path else " | Filled"
    name += " | Pretrained" if pretrained else " | Cold Start"

    run_args = dict(run_name=name)
    if pretrained:
        start_e = checkpoint['epoch'] + 1
        run_metrics = checkpoint['run_metrics']
        run_args['run_id'] = run_id
    else:
        start_e = 1
        run_metrics = dict(
            train=dict(
                batch=list(),
                loss_total=list(),
                loss_classifier=list(),
                loss_box_reg=list(),
                loss_mask=list(),
                loss_objectness=list(),
                loss_rpn_box_reg=list()
            ),
            valid=dict(
                batch=list(),
                loss_total=list(),
                loss_classifier=list(),
                loss_box_reg=list(),
                loss_mask=list(),
                loss_objectness=list(),
                loss_rpn_box_reg=list()
            )
        )

    with mlflow.start_run(**run_args) as run:
        run_id = run.info.run_id
        print(run_id)

        if not pretrained:
            mlflow.log_param('optimizer', type(optimizer).__name__)
            mlflow.log_param('dataset', os.path.basename(dataset_path))
            for tp in tracked_params:
                try:
                    mlflow.log_param(tp, optimizer.state_dict()['param_groups'][0][tp])
                except KeyError as err:
                    pass

        for e in range(start_e, num_epochs + start_e):
            print(f'\n-----Epoch {e} started-----\n')

            since = time.time()

            mlflow.log_metric('lr', optimizer.state_dict()['param_groups'][0]['lr'], step=e)

            model.train()
            bar = tq(train_loader, desc=f"Training [{e}]",
                     postfix={
                         "loss_total": 0.,
                         "loss_classifier": 0.,
                         "loss_box_reg": 0.,
                         "loss_mask": 0.,
                         "loss_objectness": 0.,
                         "loss_rpn_box_reg": 0
                     })
            for batch, (images, targets) in enumerate(bar, 1):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with autocast():
                    loss_dict = model(images, targets)

                loss = sum(l for l in loss_dict.values())

                if not math.isfinite(loss):
                    del images, targets
                    raise RuntimeError(f"Loss is {loss}, stopping training")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                postfix = dict(loss_total=loss.item())
                run_metrics["train"]["loss_total"].append(loss.item())
                run_metrics["train"]["batch"].append(batch)

                for k, v in loss_dict.items():
                    postfix[k] = v.item()
                    run_metrics["train"][k].append(v.item())

                bar.set_postfix(ordered_dict=postfix)



            with torch.no_grad():
                bar = tq(validation_loader, desc=f"Validation [{e}]",
                         postfix={
                             "loss_total": 0.,
                             "loss_classifier": 0.,
                             "loss_box_reg": 0.,
                             "loss_mask": 0.,
                             "loss_objectness": 0.,
                             "loss_rpn_box_reg": 0
                         })
                for batch, (images, targets) in enumerate(bar, 1):
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    with autocast():
                        loss_dict = model(images, targets)

                    loss = sum(l for l in loss_dict.values())

                    if not math.isfinite(loss):
                        del images, targets
                        raise RuntimeError(f"Loss is {loss}, stopping validation")

                    postfix = dict(loss_total=loss.item())
                    run_metrics["valid"]["loss_total"].append(loss.item())
                    run_metrics["valid"]["batch"].append(batch)

                    for k, v in loss_dict.items():
                        postfix[k] = v.item()
                        run_metrics["valid"][k].append(v.item())

                    bar.set_postfix(ordered_dict=postfix)

            time_elapsed = time.time() - since
            scheduler.step()

            for k, v in run_metrics["train"].items():
                if k == "batch":
                    continue
                mlflow.log_metric("train_" + k, mean(v[(e - 1) * len(train_loader):e * len(train_loader)]), step=e)

            for k, v in run_metrics["valid"].items():
                if k == "batch":
                    continue
                mlflow.log_metric("valid_" + k, mean(v[(e - 1) * len(validation_loader):e * len(validation_loader)]),
                                  step=e)

            state_dict = {
                'epoch': e,
                'run_id': run_id,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'run_metrics': run_metrics
            }

            mlflow.pytorch.log_state_dict(state_dict, artifact_path="checkpoint")

            checkpoint_path = f"blobs/CraterRCNN_{run_id}.pth"
            # torch.save(
            #     state_dict,
            #     checkpoint_path
            # )

            model.eval()
            with torch.no_grad():
                images, targets = next(iter(test_loader))
                images = list(image.cuda() for image in images)
                targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

                out = model(images)

            min_score = 0.7

            boxes, labels, scores, masks = map(lambda x: x.cpu(), out[0].values())

            fig, axes = plt.subplots(1, 4, figsize=(25, 20))

            axes[0].imshow(images[0][0].cpu().numpy(), cmap='gray')
            axes[1].imshow(torch.sum(targets[0]['masks'], dim=0).clamp(0, 1).cpu().numpy(), cmap='gray')
            axes[2].imshow(torch.sum(out[0]['masks'][scores > min_score], dim=0).clamp(0, 1).cpu().numpy()[0],
                           cmap='gray')
            draw_patches(images[0].cpu(), boxes, labels, scores, masks, min_score=min_score, ax=axes[3])

            mlflow.log_figure(fig, f"sample_output_e{e}.png")

            print(
                f"\nSummary:\n",
                f"\tEpoch: {e}/{num_epochs + start_e}\n",
                f"\tAverage train loss: {mean(run_metrics['train']['loss_total'][(e - 1) * len(train_loader):e * len(train_loader)])}\n",
                f"\tAverage validation loss: {mean(run_metrics['valid']['loss_total'][(e - 1) * len(validation_loader):e * len(validation_loader)])}\n",
                f"\tDuration: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
            print(f'-----Epoch {e} finished.-----\n')
