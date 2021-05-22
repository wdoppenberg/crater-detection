import torch

from src import CraterDetector


if __name__ == "__main__":
    model = CraterDetector()
    model.load_state_dict(torch.load("blobs/CraterRCNN.pth"))

