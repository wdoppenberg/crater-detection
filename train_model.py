import argparse

from src.detection import create_detection_model, train_model


def get_args():
    parser = argparse.ArgumentParser(description='Train the crater detection model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=12,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=10,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-n', '--numworkers', dest='num_workers', type=int, default=4,
                        help='Number of workers for training dataloader')
    parser.add_argument('-l', '--learning_rate', metavar='LR', type=float, default=0.001,
                        help='Learning rate', dest='lr', nargs='?')
    parser.add_argument('-bb', '--backbone', dest='backbone_name', type=str, default="resnet18",
                        help='Model backbone ResNet type.')
    parser.add_argument('-i', '--runid', dest='run_id', type=str, default=None, nargs='?',
                        help='Resume from MLflow run checkpoint')
    parser.add_argument('-d', '--dataset', dest='dataset_path', type=str, default="data/dataset_instanced_edge.h5",
                        help='Dataset path')
    parser.add_argument('-m', '--momentum', dest='momentum', type=float, default=0.5,
                        help='Momentum input for SGD optimizer.')
    parser.add_argument('-w', '--weightdecay', dest='weight_decay', type=float, default=1e-5,
                        help='Weight decay input for SGD optimizer.')
    parser.add_argument('-device', '--device', dest='device', type=str, default='cuda',
                        help='Device to train model on (`cpu` or `cuda`)')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    model = create_detection_model(args.backbone_name)

    train_model(model,
                num_epochs=args.epochs,
                dataset_path=args.dataset_path,
                initial_lr=args.lr,
                run_id=args.run_id,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                device=args.device
                )
