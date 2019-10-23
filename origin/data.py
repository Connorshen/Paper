import h5py
import numpy as np
import pickle
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader, TensorDataset
import torch

warnings.filterwarnings("ignore")


def convert_data():
    digits = np.arange(10)
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for digit in tqdm(digits):
        file_name = "../filterdata/digit{0}.mat".format(digit)
        data = h5py.File(file_name)["D_filtered"][:].T
        train_data.append(data)
        train_label.append(np.expand_dims(np.array([digit] * len(data)), axis=1))
        file_name = "../filterdata/test{0}.mat".format(digit)
        data = h5py.File(file_name)["D_filtered"][:].T
        test_data.append(data)
        test_label.append(np.expand_dims(np.array([digit] * len(data)), axis=1))
    train_data = np.vstack(train_data)
    train_label = np.squeeze(np.vstack(train_label), axis=1)
    test_data = np.vstack(test_data)
    test_label = np.squeeze(np.vstack(test_label), axis=1)
    data_dict = {"train_data": train_data,
                 "train_label": train_label,
                 "test_data": test_data,
                 "test_label": test_label}
    fw = open("../filterdata/data_all.pkl", "wb")
    pickle.dump(data_dict, fw)
    fw.close()


def loader(batch_size, shuffle):
    fr = open("../filterdata/data_all.pkl", "rb")
    data_dict = pickle.load(fr)
    train_data = data_dict["train_data"]
    train_label = data_dict["train_label"]
    test_data = data_dict["test_data"]
    test_label = data_dict["test_label"]
    train_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_label).long())
    test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_label).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader


if __name__ == '__main__':
    convert_data()
