import os

from os import path


class DataConfig:
    ROOT = path.dirname(path.realpath(__file__))
    DATA_PATH = path.join(ROOT, "preprocess")
    MNIST_PATH = path.join(DATA_PATH, "mnist")
    FASHION_PATH = path.join(DATA_PATH, "fashion_mnist")
    MNIST_TRAIN_IMAGE_PATH = path.join(MNIST_PATH, "train-images-idx3-ubyte")
    MNIST_TRAIN_LABEL_PATH = path.join(MNIST_PATH, "train-labels-idx1-ubyte")
    MNIST_TEST_IMAGE_PATH = path.join(MNIST_PATH, "t10k-images-idx3-ubyte")
    MNIST_TEST_LABEL_PATH = path.join(MNIST_PATH, "t10k-labels-idx1-ubyte")
    FASHION_TRAIN_IMAGE_PATH = path.join(FASHION_PATH, "train-images-idx3-ubyte")
    FASHION_TRAIN_LABEL_PATH = path.join(FASHION_PATH, "train-labels-idx1-ubyte")
    FASHION_TEST_IMAGE_PATH = path.join(FASHION_PATH, "t10k-images-idx3-ubyte")
    FASHION_TEST_LABEL_PATH = path.join(FASHION_PATH, "t10k-labels-idx1-ubyte")
