import numpy as np
from tqdm import tqdm
import h5py
import scipy.io as scio
import warnings

warnings.filterwarnings("ignore")
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
train_label = np.reshape(np.squeeze(np.vstack(train_label), axis=1), (-1, 1))
test_data = np.vstack(test_data)
test_label = np.reshape(np.squeeze(np.vstack(test_label), axis=1), (-1, 1))
train_dict = {"train_img": train_data, "train_label": train_label}
test_dict = {"test_img": test_data, "test_label": test_label}
scio.savemat("train_origin.mat", train_dict)
scio.savemat("test_origin.mat", test_dict)
