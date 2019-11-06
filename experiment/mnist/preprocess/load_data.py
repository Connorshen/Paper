import numpy as np
from struct import unpack
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

train_image_path = "train-images-idx3-ubyte"
train_label_path = "train-labels-idx1-ubyte"
test_image_path = "t10k-images-idx3-ubyte"
test_label_path = "t10k-labels-idx1-ubyte"


def __read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img


def __read_label(path):
    with open(path, 'rb') as f:
        _, _ = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab


def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img


def __one_hot_label(label):
    lab = np.zeros((label.size, 10), dtype=np.uint8)
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def load_file(train_image_path,
              train_label_path,
              test_image_path,
              test_label_path,
              normalize=True,
              one_hot=True):
    """
    :param train_image_path:
    :param train_label_path:
    :param test_image_path:
    :param test_label_path:
    :param normalize: 是否规范化
    :param one_hot: 是否独热码
    :return:
    """
    image = {
        'train': __read_image(train_image_path),
        'test': __read_image(test_image_path)
    }

    label = {
        'train': __read_label(train_label_path),
        'test': __read_label(test_label_path)
    }

    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return image['train'], label['train'], image['test'], label['test']


def load_mnist(flatten=True, one_hot=True, ratio=1.0, digits=np.arange(0, 10)):
    train_image, train_label, test_image, test_label = load_file(train_image_path,
                                                                 train_label_path,
                                                                 test_image_path,
                                                                 test_label_path,
                                                                 one_hot=one_hot)
    if flatten is not True:
        train_image = train_image.reshape(-1, 1, 28, 28, )
        test_image = test_image.reshape(-1, 1, 28, 28, )
    if ratio != 1:
        train_image, _, train_label, _ = train_test_split(train_image, train_label, train_size=ratio)
        test_image, _, test_label, _ = train_test_split(test_image, test_label, train_size=ratio)
    if one_hot is False:
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        for digit in digits:
            train_index = train_label == digit
            train_images.append(train_image[train_index])
            train_labels.append(np.reshape(train_label[train_index], [-1, 1]))
            test_index = test_label == digit
            test_images.append(test_image[test_index])
            test_labels.append(np.reshape(test_label[test_index], [-1, 1]))
        train_image = np.vstack(train_images)
        train_label = np.reshape(np.vstack(train_labels), [-1])
        test_image = np.vstack(test_images)
        test_label = np.reshape(np.vstack(test_labels), [-1])
    return train_image, train_label, test_image, test_label


def mnist_loader(batch_size=32, shuffle=True, flatten=True, one_hot=False, digits=np.arange(0, 10)):
    train_image, train_label, test_image, test_label = load_mnist(flatten=flatten, one_hot=one_hot, digits=digits)
    train_dataset = TensorDataset(torch.from_numpy(train_image), torch.from_numpy(train_label).long())
    test_dataset = TensorDataset(torch.from_numpy(test_image), torch.from_numpy(test_label).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
