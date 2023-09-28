import torch
import sys
import importlib
import numpy as np
import os
sys.path.append("../")
import subprocess

import Model.RFAE_model as model

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





p_epochs_number = 1000
p_batch_size_value = 128

device = get_free_gpu()
p_key_feature_number = 50
p_seed = 0
#   Parameter settings

file_path = "./Data/mnist.npz"      # File path, location to store the data
datasetname = "MNIST"


dataset = np.load(file_path)
x_train_, y_train_ = dataset['x_train'], dataset['y_train']
x_test_, y_test_ = dataset['x_test'], dataset['y_test']
x_data = np.r_[x_train_, x_test_].reshape(70000, 28 * 28).astype('float32') / 255.0
y_data = np.r_[y_train_, y_test_]
#    read data

clf = True

clf_feature_list = model.cal(x_data, y_data, datasetname, p_key_feature_number, p_epochs_number, p_batch_size_value, clf, p_seed, device)
print(clf_feature_list)
clf = False

feature_list = model.cal(x_data, y_data, datasetname, p_key_feature_number, p_epochs_number, p_batch_size_value, clf, p_seed, device)
print(feature_list)