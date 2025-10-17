"""
Read data from mat files and save them into training and test npy files

- reorder the channels
- reorder the trials
- standardize the data
- baseline correction? currently no
"""
import os
import scipy.io
import scipy.signal as signal
import numpy as np
import pandas as pd
import h5py
import pickle
import matplotlib.pyplot as plt
from einops import rearrange

data_path = '/media/siat/disk1/BCI_data/THINGS-MEG/derivatives/preprocessed/'
label_path = '/media/siat/disk1/BCI_data/THINGS-MEG/label_csv/'
save_path = '/media/siat/disk1/BCI_data/THINGS-MEG/derivatives/preprocessed_divided/'


for sub_idx in range(4):
    sub_idx += 1
    # read matlab file with h5py
    data_mat = h5py.File(data_path + 'P{:01d}_cosmofile.mat'.format(sub_idx), 'r')
    data_mat = data_mat['ds']

    data = np.array(data_mat['samples'])

    data = rearrange(data, '(time channels) trials -> trials channels time', channels=271, time=281)

    del data_mat

    # read label file with csv
    pd_label = pd.read_csv(label_path + 'sample_attributes_P{:01d}.csv'.format(sub_idx))
    trial_type = pd_label['trial_type'].values
    things_category = pd_label['things_category_nr'].values # float 64
    things_exemplar_nr = pd_label['things_exemplar_nr'].values
    test_image = pd_label['test_image_nr'].values
    image_path = pd_label['image_path'].values

    train_idx = np.sort(np.where(trial_type == 'exp')[0])
    test_idx = np.sort(np.where(trial_type == 'test')[0])
    
    print('The length of train_id is: ', len(train_idx))
    print('The length of test_id is: ', len(test_idx))
    # train_category = things_category[train_idx]
    # test_category = things_category[test_idx]

    train_data = np.zeros((22248, data.shape[1], data.shape[2])) 
    test_data = [[] for i in range(200)]
    # test_data = np.zeros((200, 12, data.shape[1], data.shape[2]))

    for trn_idx in range(22248):
        train_data[12*(np.int32(things_category[train_idx[trn_idx]])-1)+(np.int32(things_exemplar_nr[train_idx[trn_idx]])-1), :, :] = data[train_idx[trn_idx], :, :]

    for tst_idx in range(2400):
        test_data[np.int32(test_image[test_idx[tst_idx]])-1].append(data[test_idx[tst_idx]])
    test_data = np.array(test_data)
    
    # # Baseline correction
    # train_data = train_data - np.mean(train_data[:, :, :25], axis=2).reshape(-1, train_data.shape[1], 1)
    # test_data = test_data - np.mean(test_data[:, :, :, :25], axis=3).reshape(-1, test_data.shape[1], test_data.shape[2], 1)

    # segment the data into 1s
    train_data = train_data[:, :, 20:-60]
    test_data = test_data[:, :, :, 20:-60]

    # save train data
    save_pic = open(save_path + 'meg_sub-{:02d}_train.pkl'.format(sub_idx), 'wb')
    pickle.dump(train_data, save_pic, protocol=4)
    save_pic.close()
    
    # save test data
    save_pic = open(save_path + 'meg_sub-{:02d}_test.pkl'.format(sub_idx), 'wb')
    pickle.dump(test_data, save_pic, protocol=4)
    save_pic.close()

    print('Finish Subject {:02d}'.format(sub_idx))

    

