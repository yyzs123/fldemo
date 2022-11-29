from copy import deepcopy
import numpy as np
import os

def fmnist_iid_uniform(data, num_users):
    num_userdata = len(data) // num_users
    user_data = {}
    indexs = np.arange(len(data))
    np.random.shuffle(indexs)
    for i in range(num_users):
        user_data[i] = indexs[i * num_userdata : (i + 1) * num_userdata] 
    return user_data, np.linspace(num_userdata, num_userdata, num_users, dtype = int)

def fmnist_noniid_uniform(data, num_users):
    user_data = {}
    num_blocks, block_size = 2 * num_users, 30000 // num_users
    block_idxs = np.arange(num_blocks)
    np.random.shuffle(block_idxs)

    # sort data
    labels = np.array(data.targets)
    indexs = labels.argsort()

    # divide and assign
    for i in range(num_users):
        bid1, bid2 = block_idxs[i * 2], block_idxs[i * 2 + 1]
        user_data[i] = np.concatenate((indexs[bid1 * block_size : (bid1 + 1) * block_size], indexs[bid2 * block_size : (bid2 + 1) * block_size]), axis = 0)

    return user_data, np.linspace(block_size * 2 , block_size * 2, num_users, dtype = int)


def cifar10_iid_uniform(data, num_users):
    num_userdata = len(data) // num_users
    user_data = {}
    indexs = np.arange(len(data))
    np.random.shuffle(indexs)
    for i in range(num_users):
        user_data[i] = indexs[i * num_userdata : (i + 1) * num_userdata] 
    return user_data, np.linspace(num_userdata, num_userdata, num_users, dtype = int)

def cifar10_noniid_uniform(data, num_users):
    user_data = {}
    num_blocks, block_size = 2 * num_users, 25000 // num_users
    block_idxs = np.arange(num_blocks)
    np.random.shuffle(block_idxs)

    # sort data
    labels = np.array(data.targets)
    indexs = labels.argsort()

    # divide and assign
    for i in range(num_users):
        bid1, bid2 = block_idxs[i * 2], block_idxs[i * 2 + 1]
        user_data[i] = np.concatenate((indexs[bid1 * block_size : (bid1 + 1) * block_size], indexs[bid2 * block_size : (bid2 + 1) * block_size]), axis = 0)

    return user_data, np.linspace(block_size * 2 , block_size * 2, num_users, dtype = int)

def split_data(data, num_users, dataset, is_iid, is_uniform):
    if dataset == 'cifar-10':
        if is_iid == True:
            return cifar10_iid_uniform(data, num_users)
        else:
            return cifar10_noniid_uniform(data, num_users)
    if is_iid == True:
        return fmnist_iid_uniform(data, num_users)
    return fmnist_noniid_uniform(data, num_users)