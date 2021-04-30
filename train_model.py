import argparse

from src.common.data import inspect_dataset
from src.detection import create_detection_model, train_model


def get_parser():
    parser = argparse.ArgumentParser(description='Train the crater detection model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for training dataloader')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate', nargs='?')
    parser.add_argument('--backbone', type=str, default="resnet18",
                        help='Model backbone ResNet type.')
    parser.add_argument('--run_id',  type=str, default=None, nargs='?',
                        help='Resume from MLflow run checkpoint')
    parser.add_argument('--dataset', type=str, default="data/dataset_9f4bd6b6-76d1-4aea-8854-cb50f15110bb.h5",
                        help='Dataset path')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum input for SGD optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay input for SGD optimizer.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train model on (`cpu` or `cuda`)')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    data_header = inspect_dataset(args.dataset, plot=False)

    model = create_detection_model(args.backbone, image_size=data_header['resolution'])

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
