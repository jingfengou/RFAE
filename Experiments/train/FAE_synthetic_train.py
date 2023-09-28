import torch
import sys
import importlib
import numpy as np
import os
import subprocess
sys.path.append("../")


import Model.FAE_synthetic as model
import Datareader.datareader as Datareader
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
test_dataset_path = ['16visualdata']



modelname = "FAE"


p_epochs_number = 1000
p_batch_size_value = 128
p_is_use_bias = True
learning_rate = 0.001
lamda1 = 2
lamda2 = 0.1

for low_dimension_dataset_path_i in test_dataset_path:
    print(low_dimension_dataset_path_i)
    device = get_free_gpu()
    p_key_feture_number = 50
    rec_loss = []
    classify_acc = []
    p_data_arr, p_label_arr_onehot = Datareader.datareader(low_dimension_dataset_path_i)
    for p_seed in np.arange(0, 1):
        orig_train_acc, orig_test_acc, selec_train_acc, selec_test_acc, reconstruction_loss = model.cal(p_data_arr,
                                                                                                           p_label_arr_onehot,
                                                                                                           p_key_feture_number,
                                                                                                           p_epochs_number,
                                                                                                           p_batch_size_value,
                                                                                                           p_is_use_bias,
                                                                                                           p_seed,
                                                                                                           learning_rate,
                                                                                                           lamda1,
                                                                                                           lamda2,
                                                                                                           low_dimension_dataset_path_i,
                                                                                                           modelname,
                                                                                                        device
                                                                                                           )
        rec_loss.append(reconstruction_loss)
        classify_acc.append(selec_test_acc)
    rec_loss_mean = np.mean(rec_loss)
    rec_loss_var = np.var(rec_loss)

    print("reconstruction loss Mean:", rec_loss_mean)
    print("reconstruction loss Variance:", rec_loss_var)

    cla_acc_mean = np.mean(classify_acc)
    cla_acc_var = np.var(classify_acc)

    print("classify acc Mean:", cla_acc_mean)
    print("classify acc Variance:", cla_acc_var)

