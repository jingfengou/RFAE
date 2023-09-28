
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
from keras import backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Activation, Dropout, Layer

from keras.utils import to_categorical
from keras import optimizers, initializers, constraints, regularizers
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.utils import plot_model

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

import h5py
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.sparse as sparse
import sys
import Functions as F
seed=0
os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)
rn.seed(seed)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf =tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

#tf.set_random_seed(seed)
tf.compat.v1.set_random_seed(seed)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

K.set_session(sess)
#----------------------------Reproducible----------------------------------------------------------------------------------------

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
from sklearn.metrics import mean_absolute_percentage_error


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
def write_to_csv(p_data, p_path):
    directory = os.path.dirname(p_path)

    # 创建新目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    dataframe = pd.DataFrame(p_data)
    dataframe.to_csv(p_path, mode='a', header=False, index=False, sep=',')
    del dataframe

def LR_check(train, test):
    LR = LinearRegression(n_jobs=-1)
    LR.fit(train[0], train[1])
    baseline = np.zeros(test[1].shape)
    MSELR = ((LR.predict(test[0]) - test[1]) ** 2).mean()


    R2 = 1 - ((LR.predict(test[0]) - test[1]) ** 2).mean()/test[1].var()

    mape = mean_absolute_percentage_error(LR.predict(test[0]), test[1])

    pcc_mat = np.corrcoef(LR.predict(test[0]).flatten(), test[1].flatten())


    return MSELR, R2, mape, pcc_mat[0][1]
# --------------------------------------------------------------------------------------------------------------------------------
def mse_check(train, test):
    LR = LinearRegression(n_jobs=-1)
    LR.fit(train[0], train[1])
    MSELR = ((LR.predict(test[0]) - test[1]) ** 2).mean()
    return MSELR

def cal(p_data_arr, p_label_arr_onehot, p_key_feture_number, p_epochs_number, p_batch_size_value, p_is_use_bias, p_seed,
        learning_rate, lamda1, lamda2, dataset, modelname, device):
    # C_train_x, C_test_x, C_train_y, C_test_y = train_test_split(p_data_arr, p_label_arr_onehot, test_size=0.2,
    #                                                             random_state=p_seed)
    # x_train, x_validate, y_train_onehot, y_validate_onehot = train_test_split(C_train_x, C_train_y, test_size=0.1,
    #                                                                           random_state=p_seed)
    x_train = p_data_arr[:88807, :]
    x_validate = p_data_arr[88807:88807+11101, :]
    x_test = p_data_arr[(88807+11101):, :]

    x_train = x_train.astype(np.float32)
    x_validate = x_validate.astype(np.float32)
    x_test = x_test.astype(np.float32)
    feature_num = x_train.shape[1]
    os.environ['PYTHONHASHSEED'] = str(p_seed)
    np.random.seed(p_seed)
    rn.seed(p_seed)
    torch.manual_seed(p_seed)
    tf.compat.v1.set_random_seed(p_seed)

    class MyDataset(Dataset):
        def __init__(self, data, label):
            self.data = data  # 加载npy数据
            self.label = label

        def __getitem__(self, index):
            data = self.data[index, :]  # 读取每一个npy的数据
            label = self.label[index]
            data = torch.tensor(data)  # 转为tensor形式
            label = torch.tensor(label)
            return data, label

        def __len__(self):
            return self.data.shape[0]  # 返回数据的总个数

    class feature_selection(nn.Module):

        def __init__(self, feature_num) -> None:

            super(feature_selection, self).__init__()
            # self.k = k
            self.weight = torch.nn.Parameter(torch.rand(feature_num), requires_grad=True)
            w = 1 - self.weight.data / 1000000

            self.weight.data = w

        def forward(self, x: Tensor, k, is_select=False) -> Tuple[Any, Any]:

            # w = self.weight.data.clamp(0, 100)
            # self.weight.data = w
            self.topk = torch.zeros(self.weight.data.shape)
            self.topk = self.topk.to(device=try_gpu(device))

            __, idx1 = torch.sort(self.weight, descending=True)

            self.topk.index_fill_(0, idx1[:k], 1)

            w1 = self.weight

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

            loss = loss1(x, y1) + lamda1 * loss1(x, y2) + lamda2 * torch.norm(WI, p=1)

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

                train_loss, val_loss = 0, 0

                # 训练模型
                self.train()  # 切换到训练模式
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    inputs = inputs.view(inputs.size(0), feature_num)
                    # => [b, 10]
                    inputs = inputs.to(device=try_gpu(device))
                    # 模型前向传播
                    out1, out2 = self(inputs, k)

                    # 计算损失函数和反向传播
                    loss = criterion(inputs, out1, out2, self.fs.weight, lamda1, lamda2)
                    loss.backward()
                    optimizer.step()
                    self.fs.weight.data = self.fs.weight.data.clamp(0, 100)

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

            torch.save(best_model, "../" + dataset + "/" + modelname + "_" + str(p_key_feture_number) + "/log/" + str(
                modelname) + str(p_seed) + "bestmodel.pth")
            return self
    directory = os.path.dirname("../"+dataset + "/" + modelname + "_" + str(p_key_feture_number) + "/log/")

    # 创建新目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    train_dataset = MyDataset(x_train, x_train)
    # test_dataset = MyDataset(x_test, x_test)
    val_dataset = MyDataset(x_validate, x_validate)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=p_batch_size_value, shuffle=True)
    # test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=p_batch_size_value, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=p_batch_size_value, shuffle=True)
    t_start = time.time()
    MiceProtein_TFAE = model(feature_num, p_key_feture_number)
    MiceProtein_TFAE = MiceProtein_TFAE.to(device=try_gpu(device))
    MiceProtein_TFAE.fit(train_loader, val_loader, p_epochs_number, learning_rate, p_key_feture_number, lamda1, lamda2)
    t_used = time.time() - t_start

    write_to_csv(np.array([t_used]), "../"+dataset + "/" + modelname + "_" + str(p_key_feture_number) + "/log/" + str(modelname) + str(p_key_feture_number) + "_time.csv")
    # --------------------------------------------------------------------------------------------------------------------------------
    x_test_torch = torch.tensor(x_test)
    x_test_torch = x_test_torch.to(device=try_gpu(device))
    p_data1, p_data2 = MiceProtein_TFAE(x_test_torch, p_key_feture_number)
    numbers = x_test.shape[0] * x_test.shape[1]

    print("Completed on " + str(p_seed) + "!")
    print("MSE for one-to-one map layer", np.sum(np.power(np.array(p_data1.detach().cpu()) - x_test, 2)) / numbers)
    print("MSE for feature selection layer", np.sum(np.power(np.array(p_data2.detach().cpu()) - x_test, 2)) / numbers)

    key_features = F.top_k_keepWeights_1(MiceProtein_TFAE.fs.weight.data.detach().cpu().numpy(), p_key_feture_number)

    # --------------------------------------------------------------------------------------------------------------------------------
    # Identify features successfully
    # if np.sum(MiceProtein_TFAE.fs.weight.data.detach().cpu().numpy() > 0) > 0:
    # Classification on original features
    # train_feature = C_train_x
    # train_label = C_train_y
    # test_feature = C_test_x
    # test_label = C_test_y
    # orig_train_acc, orig_test_acc = F.ETree(train_feature, train_label, test_feature, test_label, 0)
    #
    # # Classification on selected features
    selected_position_list = np.where(key_features > 0)[0]
    # print(selected_position_list)
    # train_feature_ = np.multiply(C_train_x, key_features)
    # train_feature = F.compress_zero_withkeystructure(train_feature_, selected_position_list)
    # train_label = C_train_y
    #
    # test_feature_ = np.multiply(C_test_x, key_features)
    # test_feature = F.compress_zero_withkeystructure(test_feature_, selected_position_list)
    # test_label = C_test_y
    # selec_train_acc, selec_test_acc = F.ETree(train_feature, train_label, test_feature, test_label, 0)

    # Linear reconstruction
    train_feature_ = np.multiply(x_train, key_features)
    C_train_selected_x = F.compress_zero_withkeystructure(train_feature_, selected_position_list)

    test_feature_ = np.multiply(x_test, key_features)
    C_test_selected_x = F.compress_zero_withkeystructure(test_feature_, selected_position_list)

    train_feature_tuple = (C_train_selected_x, x_train)
    test_feature_tuple = (C_test_selected_x, x_test)

    reconstruction_loss = mse_check(train_feature_tuple, test_feature_tuple)
    LR_loss = LR_check(train_feature_tuple, test_feature_tuple)
    # print("Classification on original data", orig_test_acc)
    # print("Classification on selected features", selec_test_acc)
    print("Linear reconstruction loss", reconstruction_loss)

    print("Linear R2 loss", LR_loss[1])
    print("Linear MAPE loss", LR_loss[2])
    print("Linear pcc value", LR_loss[3])
    print("-----------------------------------------------------------------------------")
    print("\n\n")
    #
    # else:
    #     # orig_train_acc = -1
    #     # orig_test_acc = -1
    #     # selec_train_acc = -1
    #     # selec_test_acc = -1
    #     reconstruction_loss = -1
    #     selected_position_list = np.where(key_features > 0)[0]

    results = np.array([reconstruction_loss])
    LR_results = np.array(LR_loss)
    write_to_csv(LR_results.reshape(1, len(LR_results)),
                 "../" + dataset + "/" + modelname + "_" + str(p_key_feture_number) + "/log/LR_results.csv")
    write_to_csv(results.reshape(1, len(results)), "../"+dataset + "/" + modelname + "_" + str(p_key_feture_number) + "/log/" + str(modelname) + str(p_key_feture_number) + "_results.csv")
    write_to_csv(selected_position_list.reshape(1, len(selected_position_list)), "../"+dataset + "/" + modelname + "_" + str(p_key_feture_number) + "/log/"+ str(modelname) + str(p_key_feture_number) + "_selected_list.csv")
    return reconstruction_loss