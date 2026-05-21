import argparse
import gzip
import pickle
from pathlib import Path
from struct import unpack

import numpy as np

import mynn as nn


def load_mnist(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        _, num, _, _ = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(labels_path, 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images.astype(np.float32) / 255.0, labels


def infer_model_type(model_path):
    with open(model_path, 'rb') as f:
        state = pickle.load(f)
    if isinstance(state, dict) and state.get('model_type') == 'Model_CNN':
        return 'cnn'
    return 'mlp'


def load_model(model_path, model_type):
    if model_type == 'auto':
        model_type = infer_model_type(model_path)
    if model_type == 'cnn':
        model = nn.models.Model_CNN()
    else:
        model = nn.models.Model_MLP()
    model.load_model(model_path)
    return model


def evaluate(model, images, labels, batch_size):
    total_correct = 0
    total_count = 0
    for start in range(0, images.shape[0], batch_size):
        batch_X = images[start:start + batch_size]
        batch_y = labels[start:start + batch_size]
        logits = model(batch_X)
        total_correct += (np.argmax(logits, axis=-1) == batch_y).sum()
        total_count += batch_X.shape[0]
    return total_correct / total_count


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a saved MNIST model.')
    parser.add_argument('--model-path', default='./best_models/mlp_sgd/best_model.pickle')
    parser.add_argument('--model-type', choices=['auto', 'mlp', 'cnn'], default='auto')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--images-path', default='./dataset/MNIST/t10k-images-idx3-ubyte.gz')
    parser.add_argument('--labels-path', default='./dataset/MNIST/t10k-labels-idx1-ubyte.gz')
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    images, labels = load_mnist(args.images_path, args.labels_path)
    model = load_model(model_path, args.model_type)
    accuracy = evaluate(model, images, labels, args.batch_size)
    print(f'test_accuracy={accuracy:.6f}')


if __name__ == '__main__':
    main()
