import argparse

from src.detection import CraterDetector, train_model
from src.common.data import inspect_dataset


def get_parser():
    parser = argparse.ArgumentParser(description='Train the crater detection model on images and target ellipses',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for training dataloader')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate', nargs='?')
    parser.add_argument('--backbone', type=str, default="resnet50",
                        choices=['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                 'resnext50_32x4d', 'resnext101_32x8d',
                                 'wide_resnet50_2', 'wide_resnet101_2'], help='Model backbone ResNet type.')
    parser.add_argument('--run_id',  type=str, default=None, nargs='?',
                        help='Resume from MLflow run checkpoint')
    parser.add_argument('--dataset', type=str, default="data/dataset_crater_detection_80k.h5",
                        help='Dataset path')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum input for SGD optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay input for SGD optimizer.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train model on (`cpu` or `cuda`)')
    parser.add_argument('--ellipse_loss_metric', type=str, default='gaussian-angle',
                        choices=['gaussian-angle', 'kullback-leibler'], help='Ellipse loss metric for EllipseRoIHeads.')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    data_header = inspect_dataset(args.dataset, plot=False)

    model = CraterDetector(backbone_name=args.backbone, ellipse_loss_metric=args.ellipse_loss_metric)

    train_model(model,
                num_epochs=args.epochs,
                dataset_path=args.dataset,
                initial_lr=args.learning_rate,
                run_id=args.run_id,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                device=args.device
                )
