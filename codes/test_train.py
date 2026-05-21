import argparse
import gzip
import json
from pathlib import Path
from struct import unpack

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from draw_tools.plot import plot


def load_mnist(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        _, num, _, _ = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(labels_path, 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    images = images.astype(np.float32) / 255.0
    return images, labels


def build_model(args, input_dim, num_classes):
    lambdas = [args.weight_decay, args.weight_decay]
    if args.model == 'mlp':
        return nn.models.Model_MLP([input_dim, args.hidden_dim, num_classes], 'ReLU', lambdas)
    return nn.models.Model_CNN(
        input_shape=(1, 28, 28),
        num_classes=num_classes,
        conv_channels=args.conv_channels,
        kernel_size=3,
        stride=args.conv_stride,
        padding=1,
        lambda_list=lambdas,
    )


def build_optimizer(args, model):
    if args.optimizer == 'momentum':
        return nn.optimizer.MomentGD(init_lr=args.lr, model=model, mu=args.momentum)
    return nn.optimizer.SGD(init_lr=args.lr, model=model)


def build_scheduler(args, optimizer):
    if args.scheduler == 'multistep':
        milestones = [int(item) for item in args.milestones.split(',') if item.strip()]
        return nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=args.gamma)
    if args.scheduler == 'exp':
        return nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)
    return None


def save_curve(runner, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.set_tight_layout(True)
    plot(runner, axes)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP or CNN on MNIST.')
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--optimizer', choices=['sgd', 'momentum'], default='sgd')
    parser.add_argument('--scheduler', choices=['none', 'multistep', 'exp'], default='multistep')
    parser.add_argument('--lr', type=float, default=0.06)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=600)
    parser.add_argument('--conv-channels', type=int, default=8)
    parser.add_argument('--conv-stride', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--log-iters', type=int, default=100)
    parser.add_argument('--valid-size', type=int, default=10000)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--milestones', default='800,2400,4000')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=309)
    parser.add_argument('--save-dir', default='best_models')
    parser.add_argument('--figure-dir', default='figs')
    parser.add_argument('--run-name', default='')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    train_images_path = Path('./dataset/MNIST/train-images-idx3-ubyte.gz')
    train_labels_path = Path('./dataset/MNIST/train-labels-idx1-ubyte.gz')
    images, labels = load_mnist(train_images_path, train_labels_path)

    idx = np.random.permutation(np.arange(images.shape[0]))
    images = images[idx]
    labels = labels[idx]

    valid_imgs = images[:args.valid_size]
    valid_labs = labels[:args.valid_size]
    train_imgs = images[args.valid_size:]
    train_labs = labels[args.valid_size:]

    if args.train_limit > 0:
        train_imgs = train_imgs[:args.train_limit]
        train_labs = train_labs[:args.train_limit]

    model = build_model(args, input_dim=train_imgs.shape[-1], num_classes=int(labels.max()) + 1)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=int(labels.max()) + 1)

    run_name = args.run_name or f'{args.model}_{args.optimizer}'
    save_dir = Path(args.save_dir) / run_name
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    runner = nn.runner.RunnerM(
        model,
        optimizer,
        nn.metric.accuracy,
        loss_fn,
        batch_size=args.batch_size,
        scheduler=scheduler,
    )
    runner.train(
        [train_imgs, train_labs],
        [valid_imgs, valid_labs],
        num_epochs=args.num_epochs,
        log_iters=args.log_iters,
        save_dir=str(save_dir),
    )

    curve_path = figure_dir / f'{run_name}_curve.png'
    save_curve(runner, curve_path)

    summary = {
        'model': args.model,
        'run_name': run_name,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'train_size': int(train_imgs.shape[0]),
        'valid_size': int(valid_imgs.shape[0]),
        'best_valid_accuracy': float(runner.best_score),
        'curve': str(curve_path),
        'checkpoint': str(save_dir / 'best_model.pickle'),
    }
    summary_path = save_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
