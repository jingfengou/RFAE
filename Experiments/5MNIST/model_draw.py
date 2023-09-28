import torch

import random as rn
import os

from torch import nn

from torch import optim
import math
from typing import Any, Tuple

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import numpy as np
import tensorflow as tf
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import torch._dynamo
import time

seed = 0
dataset = 'MNIST'
modelname = 'TFAEexp'

data = np.load('./Dataset/mnist.npz')
num_data_used = 1
x_train_, y_train_ = data['x_train'], data['y_train']
x_test_, y_test_ = data['x_test'], data['y_test']
x_data = np.r_[x_train_, x_test_].reshape(70000, 28 * 28).astype('float32') / 255.0
y_data = np.r_[y_train_, y_test_]
np.random.seed(seed)
x_data_num, _ = x_data.shape
index = np.arange(x_data_num)
np.random.shuffle(index)

data_arr = x_data[index][0:num_data_used]
label_arr_onehot = y_data[index][0:num_data_used]

feature_num = 784


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
class feature_selection(nn.Module):

    def __init__(self, feature_num) -> None:

        super(feature_selection, self).__init__()
        # self.k = k
        self.weight = torch.nn.Parameter(torch.rand(feature_num), requires_grad=True)
        w = self.weight.data / 1000000

        self.weight.data = w

    def forward(self, x: Tensor, k, is_select=False) -> Tuple[Any, Any]:

        # w = self.weight.data.clamp(0, 100)
        # self.weight.data = w
        self.topk = torch.zeros(self.weight.data.shape)
        self.topk = self.topk.to(device=try_gpu(device))

        __, idx1 = torch.sort(self.weight, descending=True)

        self.topk.index_fill_(0, idx1[:k], 1)

        w1 = torch.exp(self.weight)

        w2 = self.topk * w1

        if is_select:
            x1 = x * w2
        else:
            x1 = x * w1

        return x1


class encoder(nn.Module):

    def __init__(self, feature_num, k):
        super(encoder, self).__init__()

        # xw+b
        self.fc1 = nn.Linear(feature_num, k)
        # self.fc2 = nn.Linear(256, 50)

        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        # x:[b, 1, 28, 28]

        # x1 = F.relu(self.fc1(x1))

        x = self.fc1(x)
        # x2 = F.relu(self.fc1(x2))

        return x


class decoder(nn.Module):

    def __init__(self, feature_num, k):
        super(decoder, self).__init__()

        # xw+b
        self.fc1 = nn.Linear(k, feature_num)
        # self.fc2 = nn.Linear(256, 28*28)

        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        # x:[b, 1, 28, 28]

        # x1 = F.relu(self.fc1(x1))
        x = self.fc1(x)
        # x2 = F.relu(self.fc1(x2))

        return x


class FractalLoss(nn.Module):
    def __init__(self):
        super(FractalLoss, self).__init__()

    def forward(self, x, y1, y2, WI, lamda1, lamda2):
        # loss1 = torch.nn.MSELoss()
        loss1 = torch.nn.MSELoss()

        loss = loss1(x, y1) + lamda1 * loss1(x, y2) + lamda2 * torch.norm(torch.exp(WI), p=1)

        return loss


class model(nn.Module):
    def __init__(self, feature_num, k):
        super(model, self).__init__()
        self.fs = feature_selection(feature_num)
        self.Encoder = encoder(feature_num, k)
        self.Decoder = decoder(feature_num, k)
        self.feature_num = feature_num

    def forward(self, x, k):
        x1 = self.fs(x, k)
        x2 = self.fs(x, k, is_select=True)
        y1 = self.Encoder(x1)
        y2 = self.Encoder(x2)

        out1 = self.Decoder(y1)
        out2 = self.Decoder(y2)
        return out1, out2

    def fit(self, train_loader, val_loader, num_epochs, learning_rate, k, lamda1, lamda2):
        # 定义损失函数和优化器
        criterion = FractalLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        N = num_epochs // 2
        best_val_loss = float('inf')  # 保存最佳验证集损失值

        for epoch in range(num_epochs):
            if epoch < N:
                ktemp = int(self.feature_num - (self.feature_num - k) * epoch / N)
            else:
                ktemp = k
            train_loss, val_loss = 0, 0

            # 训练模型
            self.train()  # 切换到训练模式
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = inputs.view(inputs.size(0), feature_num)
                # => [b, 10]
                inputs = inputs.to(device=try_gpu(device))
                # 模型前向传播
                out1, out2 = self(inputs, ktemp)

                # 计算损失函数和反向传播
                loss = criterion(inputs, out1, out2, self.fs.weight, lamda1, lamda2)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 使用验证集评估模型
            self.eval()  # 切换到评估模式
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.view(inputs.size(0), feature_num)

                    inputs = inputs.to(device=try_gpu(device))
                    out1, out2 = self(inputs, k)
                    val_loss += criterion(inputs, out1, out2, self.fs.weight, lamda1, lamda2).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(self.state_dict(), "../" + dataset + "/" + modelname + "_" + str(
                    p_key_feture_number) + "/log/tempbestmodel.pth")
            if epoch % 100 == 0:
                print("Epoch {}/{} : Train Loss {:.4f} / Val Loss {:.4f}".format(epoch + 1, num_epochs, train_loss,
                                                                                 val_loss))

        best_model = torch.load(
            "../" + dataset + "/" + modelname + "_" + str(p_key_feture_number) + "/log/tempbestmodel.pth")

        self.load_state_dict(best_model)

        torch.save(best_model,
                   "../" + dataset + "/" + modelname + "_" + str(p_key_feture_number) + "/log/" + str(modelname) + str(
                       p_seed) + "bestmodel.pth")
        return self