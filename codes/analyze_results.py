import gzip
import json
from pathlib import Path
from struct import unpack

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from mynn.op import softmax


def load_mnist(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        _, num, _, _ = unpack('>4I', f.read(16))
        raw_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(labels_path, 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return raw_images, raw_images.astype(np.float32) / 255.0, labels


def load_saved_model(model_path, model_type):
    if model_type == 'cnn':
        model = nn.models.Model_CNN()
    else:
        model = nn.models.Model_MLP()
    model.load_model(model_path)
    return model


def predict_batches(model, images, batch_size=256):
    logits_parts = []
    for start in range(0, images.shape[0], batch_size):
        logits_parts.append(model(images[start:start + batch_size]))
    logits = np.vstack(logits_parts)
    return logits, np.argmax(logits, axis=1)


def confusion_matrix(labels, preds, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for label, pred in zip(labels, preds):
        matrix[int(label), int(pred)] += 1
    return matrix


def plot_confusion_matrix(matrix, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    threshold = matrix.max() * 0.55
    for i in range(10):
        for j in range(10):
            color = 'white' if matrix[i, j] > threshold else 'black'
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_misclassified(raw_images, labels, preds, logits, title, save_path, count=16):
    probabilities = softmax(logits)
    wrong_indices = np.where(preds != labels)[0]
    if wrong_indices.size == 0:
        return []
    confidence = probabilities[wrong_indices, preds[wrong_indices]]
    selected = wrong_indices[np.argsort(-confidence)[:count]]

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(title)
    for ax, idx in zip(axes.reshape(-1), selected):
        ax.imshow(raw_images[idx].reshape(28, 28), cmap='gray_r')
        ax.set_title(f"T:{labels[idx]} P:{preds[idx]} C:{probabilities[idx, preds[idx]]:.2f}", fontsize=8)
        ax.axis('off')
    for ax in axes.reshape(-1)[len(selected):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return selected.tolist()


def plot_mlp_weights(model, save_path, count=40):
    weights = model.layers[0].params['W']
    fig, axes = plt.subplots(4, 10, figsize=(12, 5))
    fig.suptitle('MLP first-layer weights')
    for i, ax in enumerate(axes.reshape(-1)):
        image = weights[:, i].reshape(28, 28)
        limit = np.max(np.abs(image))
        ax.imshow(image, cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_cnn_kernels(model, save_path):
    kernels = model.layers[0].params['W']
    count = kernels.shape[0]
    cols = min(8, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes = np.array(axes).reshape(-1)
    fig.suptitle('CNN first-layer kernels')
    for i, ax in enumerate(axes):
        if i < count:
            image = kernels[i, 0]
            limit = np.max(np.abs(image))
            ax.imshow(image, cmap='RdBu_r', vmin=-limit, vmax=limit)
            ax.set_title(f'K{i}', fontsize=8)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def top_confusions(matrix, top_k=6):
    pairs = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and matrix[i, j] > 0:
                pairs.append((int(matrix[i, j]), int(i), int(j)))
    pairs.sort(reverse=True)
    return [{'count': count, 'true': true, 'predicted': pred} for count, true, pred in pairs[:top_k]]


def main():
    figs_dir = Path('figs')
    figs_dir.mkdir(exist_ok=True)

    raw_images, images, labels = load_mnist(
        './dataset/MNIST/t10k-images-idx3-ubyte.gz',
        './dataset/MNIST/t10k-labels-idx1-ubyte.gz',
    )

    experiments = {
        'mlp': {
            'type': 'mlp',
            'path': './best_models/mlp_baseline/best_model.pickle',
            'title': 'MLP baseline',
        },
        'cnn': {
            'type': 'cnn',
            'path': './best_models/cnn_baseline/best_model.pickle',
            'title': 'CNN baseline',
        },
        'cnn_momentum': {
            'type': 'cnn',
            'path': './best_models/cnn_momentum/best_model.pickle',
            'title': 'CNN with momentum',
        },
    }

    summary = {}
    for name, config in experiments.items():
        model = load_saved_model(config['path'], config['type'])
        logits, preds = predict_batches(model, images)
        matrix = confusion_matrix(labels, preds)
        accuracy = float(np.trace(matrix) / np.sum(matrix))
        summary[name] = {
            'accuracy': accuracy,
            'top_confusions': top_confusions(matrix),
        }
        plot_confusion_matrix(
            matrix,
            f"{config['title']} confusion matrix, acc={accuracy:.4f}",
            figs_dir / f'{name}_confusion_matrix.png',
        )
        plot_misclassified(
            raw_images,
            labels,
            preds,
            logits,
            f"{config['title']}: confident mistakes",
            figs_dir / f'{name}_misclassified.png',
        )

        if name == 'mlp':
            plot_mlp_weights(model, figs_dir / 'mlp_first_layer_weights.png')
        if name == 'cnn':
            plot_cnn_kernels(model, figs_dir / 'cnn_first_layer_kernels.png')

    summary_path = figs_dir / 'analysis_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
