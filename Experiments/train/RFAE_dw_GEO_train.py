import sys
import importlib
import numpy as np
import os
sys.path.append("../")
import subprocess
import torch

def get_free_gpu():
    # 获取可用的 GPU 设备数量
    device_count = torch.cuda.device_count()

    if device_count == 1:
        return 0  # 如果只有一个 GPU 设备，则返回设备号0

    # 获取每个 GPU 设备的显存
    # memory_usage = []
    # for i in range(device_count):
    #     with torch.cuda.device(i):
    #         allocated = torch.cuda.memory_allocated()
    #         memory_usage.append((i, allocated))

    # 使用 nvidia-smi 命令获取每个 GPU 设备的显存
    memory_usage = []
    for i in range(device_count):
        query = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id={i}"
        output = subprocess.check_output(query.split())
        allocated = int(output)
        memory_usage.append((i, allocated))

    # 排序获取未被使用的 GPU 设备
    sorted_device = sorted(memory_usage, key=lambda item: item[1])
    return sorted_device[0][0]

import Model.RFAE_dw_GEO as model
import Datareader.datareader as Datareader

gene_dataset_path = ['15GEO']

modelname = "RFAE_dw"

p_epochs_number = 200
p_batch_size_value = 256
p_is_use_bias = True
learning_rate = 0.001
lamda1 = 2
lamda2 = 0.1

for gene_dataset_path_i in gene_dataset_path:
    device = get_free_gpu()

    p_key_feture_number = 943
    rec_loss = []
    p_data_arr, p_label_arr_onehot = Datareader.datareader(gene_dataset_path_i)
    for p_seed in np.arange(0, 5):
        reconstruction_loss = model.cal(p_data_arr,
                                       p_label_arr_onehot,
                                       p_key_feture_number,
                                       p_epochs_number,
                                       p_batch_size_value,
                                       p_is_use_bias,
                                       p_seed,
                                       learning_rate,
                                       lamda1,
                                       lamda2,
                                       gene_dataset_path_i,
                                       modelname,
                                       device
                                       )
        rec_loss.append(reconstruction_loss)

    rec_loss_mean = np.mean(rec_loss)
    rec_loss_var = np.var(rec_loss)

    print("reconstruction loss Mean:", rec_loss_mean)
    print("reconstruction loss Variance:", rec_loss_var)
    device = get_free_gpu()
    p_key_feture_number = 900
    rec_loss = []
    p_data_arr, p_label_arr_onehot = Datareader.datareader(gene_dataset_path_i)
    for p_seed in np.arange(0, 5):
        reconstruction_loss = model.cal(p_data_arr,
                                       p_label_arr_onehot,
                                       p_key_feture_number,
                                       p_epochs_number,
                                       p_batch_size_value,
                                       p_is_use_bias,
                                       p_seed,
                                       learning_rate,
                                       lamda1,
                                       lamda2,
                                       gene_dataset_path_i,
                                       modelname,
                                       device
                                       )
        rec_loss.append(reconstruction_loss)
    rec_loss_mean = np.mean(rec_loss)
    rec_loss_var = np.var(rec_loss)
    print("reconstruction loss Mean:", rec_loss_mean)
    print("reconstruction loss Variance:", rec_loss_var)

    device = get_free_gpu()
    p_key_feture_number = 800
    rec_loss = []
    p_data_arr, p_label_arr_onehot = Datareader.datareader(gene_dataset_path_i)
    for p_seed in np.arange(0, 5):
        reconstruction_loss = model.cal(p_data_arr,
                                       p_label_arr_onehot,
                                       p_key_feture_number,
                                       p_epochs_number,
                                       p_batch_size_value,
                                       p_is_use_bias,
                                       p_seed,
                                       learning_rate,
                                       lamda1,
                                       lamda2,
                                       gene_dataset_path_i,
                                       modelname,
                                       device
                                       )
        rec_loss.append(reconstruction_loss)
    rec_loss_mean = np.mean(rec_loss)
    rec_loss_var = np.var(rec_loss)
    print("reconstruction loss Mean:", rec_loss_mean)
    print("reconstruction loss Variance:", rec_loss_var)

    device = get_free_gpu()
    p_key_feture_number = 700
    rec_loss = []
    p_data_arr, p_label_arr_onehot = Datareader.datareader(gene_dataset_path_i)
    for p_seed in np.arange(0, 5):
        reconstruction_loss = model.cal(p_data_arr,
                                       p_label_arr_onehot,
                                       p_key_feture_number,
                                       p_epochs_number,
                                       p_batch_size_value,
                                       p_is_use_bias,
                                       p_seed,
                                       learning_rate,
                                       lamda1,
                                       lamda2,
                                       gene_dataset_path_i,
                                       modelname,
                                        device
                                       )
        rec_loss.append(reconstruction_loss)
    rec_loss_mean = np.mean(rec_loss)
    rec_loss_var = np.var(rec_loss)
    print("reconstruction loss Mean:", rec_loss_mean)
    print("reconstruction loss Variance:", rec_loss_var)

    device = get_free_gpu()
    p_key_feture_number = 600
    rec_loss = []
    p_data_arr, p_label_arr_onehot = Datareader.datareader(gene_dataset_path_i)
    for p_seed in np.arange(0, 5):
        reconstruction_loss = model.cal(p_data_arr,
                                       p_label_arr_onehot,
                                       p_key_feture_number,
                                       p_epochs_number,
                                       p_batch_size_value,
                                       p_is_use_bias,
                                       p_seed,
                                       learning_rate,
                                       lamda1,
                                       lamda2,
                                       gene_dataset_path_i,
                                       modelname,
                                        device
                                       )
        rec_loss.append(reconstruction_loss)
    rec_loss_mean = np.mean(rec_loss)
    rec_loss_var = np.var(rec_loss)
    print("reconstruction loss Mean:", rec_loss_mean)
    print("reconstruction loss Variance:", rec_loss_var)

    device = get_free_gpu()
    p_key_feture_number = 500
    rec_loss = []
    p_data_arr, p_label_arr_onehot = Datareader.datareader(gene_dataset_path_i)
    for p_seed in np.arange(0, 5):
        reconstruction_loss = model.cal(p_data_arr,
                                       p_label_arr_onehot,
                                       p_key_feture_number,
                                       p_epochs_number,
                                       p_batch_size_value,
                                       p_is_use_bias,
                                       p_seed,
                                       learning_rate,
                                       lamda1,
                                       lamda2,
                                       gene_dataset_path_i,
                                       modelname,
                                        device
                                       )
        rec_loss.append(reconstruction_loss)
    rec_loss_mean = np.mean(rec_loss)
    rec_loss_var = np.var(rec_loss)
    print("reconstruction loss Mean:", rec_loss_mean)
    print("reconstruction loss Variance:", rec_loss_var)

