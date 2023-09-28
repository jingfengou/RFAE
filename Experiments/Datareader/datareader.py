import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import gzip
import scipy.io

seed = 0
def datareader(datasetname):
    if datasetname == '1MiceProtein' :
        data_frame = pd.read_excel('../1MiceProtein/Dataset/Data_Cortex_Nuclear.xls', sheet_name='Hoja1')

        data_arr = (np.array(data_frame)[:, 1:78]).copy()
        label_arr = (np.array(data_frame)[:, 81]).copy()

        for index_i in np.arange(len(label_arr)):
            if label_arr[index_i] == 'c-CS-s':
                label_arr[index_i] = '0'
            if label_arr[index_i] == 'c-CS-m':
                label_arr[index_i] = '1'
            if label_arr[index_i] == 'c-SC-s':
                label_arr[index_i] = '2'
            if label_arr[index_i] == 'c-SC-m':
                label_arr[index_i] = '3'
            if label_arr[index_i] == 't-CS-s':
                label_arr[index_i] = '4'
            if label_arr[index_i] == 't-CS-m':
                label_arr[index_i] = '5'
            if label_arr[index_i] == 't-SC-s':
                label_arr[index_i] = '6'
            if label_arr[index_i] == 't-SC-m':
                label_arr[index_i] = '7'

        label_arr_onehot = label_arr  # to_categorical(label_arr)

        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(data_arr)
        data_arr = imp_mean.transform(data_arr)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data_arr)
        return data_arr, label_arr_onehot
    elif datasetname == '2COIL' :
        dataset_path = '../2COIL/Dataset/coil-20-proc/'

        samples = {}
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            # print(dirpath)
            # print(dirnames)
            # print(filenames)
            dirnames.sort()
            filenames.sort()
            for filename in [f for f in filenames if f.endswith(".png") and not f.find('checkpoint') > 0]:
                full_path = os.path.join(dirpath, filename)
                file_identifier = filename.split('__')[0][3:]
                if file_identifier not in samples.keys():
                    samples[file_identifier] = []
                # Direct read
                # image = io.imread(full_path)
                # Resize read
                image_ = Image.open(full_path).resize((20, 20), Image.ANTIALIAS)
                image = np.asarray(image_)
                samples[file_identifier].append(image)

        # plt.imshow(samples['1'][0].reshape(20,20))
        # %%
        data_arr_list = []
        label_arr_list = []
        for key_i in samples.keys():
            key_i_for_label = [int(key_i) - 1]
            data_arr_list.append(np.array(samples[key_i]))
            label_arr_list.append(np.array(72 * key_i_for_label))

        data_arr = np.concatenate(data_arr_list).reshape(1440, 20 * 20).astype('float32') / 255.
        label_arr_onehot = np.concatenate(label_arr_list)  # to_categorical(np.concatenate(label_arr_list))
        return data_arr, label_arr_onehot
    elif datasetname == '3Activity':
        train_data_arr = np.array(pd.read_csv('../3Activity/Dataset/final_X_train.txt', header=None))
        test_data_arr = np.array(pd.read_csv('../3Activity/Dataset/final_X_test.txt', header=None))
        train_label_arr = (np.array(pd.read_csv('../3Activity/Dataset/final_y_train.txt', header=None)) - 1)
        test_label_arr = (np.array(pd.read_csv('../3Activity/Dataset/final_y_test.txt', header=None)) - 1)

        data_arr = np.r_[train_data_arr, test_data_arr]
        label_arr = np.r_[train_label_arr, test_label_arr]
        label_arr_onehot = label_arr  # to_categorical(label_arr)
        print(data_arr.shape)
        print(label_arr_onehot.shape)

        data_arr = MinMaxScaler(feature_range=(0, 1)).fit_transform(data_arr)
        return data_arr, label_arr_onehot
    elif datasetname == '4ISOLET':
        train_data_frame = np.array(pd.read_csv('../4ISOLET/Dataset/isolet1+2+3+4.data', header=None))
        test_data_frame = np.array(pd.read_csv('../4ISOLET/Dataset/isolet5.data', header=None))

        train_data_arr = (train_data_frame[:, 0:617]).copy()
        train_label_arr = ((train_data_frame[:, 617]).copy() - 1)
        test_data_arr = (test_data_frame[:, 0:617]).copy()
        test_label_arr = ((test_data_frame[:, 617]).copy() - 1)

        data_arr = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.r_[train_data_arr, test_data_arr])

        label_arr_onehot = np.r_[train_label_arr, test_label_arr]  # to_categorical(train_label_arr)
        return data_arr, label_arr_onehot
    elif datasetname == '5MNIST':
        num_data_used = 10000
        dataset = np.load('../5MNIST/Dataset/mnist.npz')
        x_train_, y_train_ = dataset['x_train'], dataset['y_train']
        x_test_, y_test_ = dataset['x_test'], dataset['y_test']
        x_data = np.r_[x_train_, x_test_].reshape(70000, 28 * 28).astype('float32') / 255.0
        y_data = np.r_[y_train_, y_test_]
        np.random.seed(seed)
        x_data_num, _ = x_data.shape
        index = np.arange(x_data_num)
        np.random.shuffle(index)

        data_arr = x_data[index][0:num_data_used]
        label_arr_onehot = y_data[index][0:num_data_used]
        return data_arr, label_arr_onehot
    elif datasetname == '6MNIST-Fashion':
        def get_fashion_mnist(path):
            files = [
                "train-labels-idx1-ubyte.gz",
                "train-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
            ]

            paths = []
            for fname in files:
                paths.append(path + fname)

            with gzip.open(paths[0], "rb") as lbpath:
                y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

            with gzip.open(paths[1], "rb") as imgpath:
                x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
                    len(y_train), 28, 28
                )

            with gzip.open(paths[2], "rb") as lbpath:
                y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

            with gzip.open(paths[3], "rb") as imgpath:
                x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
                    len(y_test), 28, 28
                )

            return (x_train, y_train), (x_test, y_test)

        np.random.seed(seed)
        path = "../6MNIST-Fashion/Dataset/"
        num_data_used = 10000
        (x_train_, y_train_), (x_test_, y_test_) = get_fashion_mnist(path)
        x_data = np.r_[x_train_, x_test_].reshape(70000, 28 * 28).astype('float32') / 255.0
        y_data = np.r_[y_train_, y_test_]

        x_data_num, _ = x_data.shape
        index = np.arange(x_data_num)
        np.random.shuffle(index)

        data_arr = x_data[index][0:num_data_used]
        label_arr_onehot = y_data[index][0:num_data_used]
        return data_arr, label_arr_onehot
    elif datasetname == '7USPS':

        data_path = "../7USPS/Dataset/USPS.mat"
        Data = scipy.io.loadmat(data_path)

        data_arr = Data['X'].astype('float32')
        label_arr = Data['Y'][:, 0] - 1

        label_arr_onehot = label_arr
        return data_arr, label_arr_onehot
    elif datasetname == '8GLIOMA':
        data_path = "../8GLIOMA/Dataset/GLIOMA.mat"
        Data = scipy.io.loadmat(data_path)

        data_arr_ = Data['X']
        label_arr = Data['Y'][:, 0]

        data_arr = MinMaxScaler(feature_range=(0, 1)).fit_transform(data_arr_)

        label_arr_onehot = label_arr
        return data_arr, label_arr_onehot
    elif datasetname == '9leukemia':
        data_path = "../9leukemia/Dataset/leukemia.mat"
        Data = scipy.io.loadmat(data_path)

        data_arr_ = Data['X']
        label_arr_ = Data['Y'][:, 0]

        label_arr_ = np.array([0 if label_arr_i < 0 else label_arr_i for label_arr_i in label_arr_])

        Data = StandardScaler().fit_transform(data_arr_)
        data_arr = Data

        label_arr = label_arr_

        label_arr_onehot = label_arr  # to_categorical(label_arr)
        return data_arr, label_arr_onehot
    elif datasetname == '10pixraw10P':
        data_path = "../10pixraw10P/Dataset/pixraw10P.mat"
        Data = scipy.io.loadmat(data_path)

        data_arr = Data['X'].astype('float32') / 255
        label_arr = Data['Y'][:, 0]
        label_arr_onehot = label_arr
        return data_arr, label_arr_onehot
    elif datasetname == '11ProstateGE':
        data_path = "../11ProstateGE/Dataset/Prostate_GE.mat"
        Data = scipy.io.loadmat(data_path)

        data_arr_ = Data['X']
        label_arr = Data['Y'][:, 0]

        Data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data_arr_)

        data_arr = Data
        label_arr_onehot = label_arr
        return data_arr, label_arr_onehot
    elif datasetname == '12warpAR10P':
        data_path = "../12warpAR10P/Dataset/warpAR10P.mat"
        Data = scipy.io.loadmat(data_path)

        data_arr_ = Data['X'].astype('float32') / 255
        label_arr = Data['Y'][:, 0]

        Data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data_arr_)

        data_arr = Data
        label_arr_onehot = label_arr
        return data_arr, label_arr_onehot
    elif datasetname == '13SMKCAN187':
        data_path = "../13SMKCAN187/Dataset/SMK_CAN_187.mat"
        Data = scipy.io.loadmat(data_path)

        data_arr = Data['X']
        label_arr = Data['Y'][:, 0]
        label_arr_onehot = label_arr
        return data_arr, label_arr_onehot
    elif datasetname == '14arcene':
        data_path = "../14arcene/Dataset/arcene.mat"
        Data = scipy.io.loadmat(data_path)

        data_arr = Data['X']
        label_arr_ = Data['Y'][:, 0]

        label_arr = np.array([0 if label_arr_i < 0 else label_arr_i for label_arr_i in label_arr_])
        data_arr = MinMaxScaler(feature_range=(0, 1)).fit_transform(data_arr)
        label_arr_onehot = label_arr
        return data_arr, label_arr_onehot
    elif datasetname == '15GEO':
        path = '../15GEO/Dataset/'
        xy_train = np.load(path + 'bgedv2_XY_tr_float32.npy')
        xy_validate = np.load(path + 'bgedv2_XY_va_float32.npy')
        xy_test = np.load(path + 'bgedv2_XY_te_float32.npy')

        xy_train = xy_train.astype(np.float32)
        xy_validate = xy_validate.astype(np.float32)
        xy_test = xy_test.astype(np.float32)
        data_arr = np.concatenate((xy_train, xy_validate, xy_test), axis=0)
        label_arr_onehot = np.concatenate((xy_train, xy_validate, xy_test), axis=0)
        return data_arr, label_arr_onehot
    elif datasetname == '15GEO01':
        path = '../15GEO01/Dataset/'
        xy_train = np.load(path + 'bgedv2_01norm_XY_tr_float32.npy')
        xy_validate = np.load(path + 'bgedv2_01norm_XY_va_float32.npy')
        xy_test = np.load(path + 'bgedv2_01norm_XY_te_float32.npy')

        xy_train = xy_train.astype(np.float32)
        xy_validate = xy_validate.astype(np.float32)
        xy_test = xy_test.astype(np.float32)
        data_arr = np.concatenate((xy_train, xy_validate, xy_test), axis=0)
        label_arr_onehot = np.concatenate((xy_train, xy_validate, xy_test), axis=0)
        return data_arr, label_arr_onehot
    elif datasetname == '16visualdata':
        path = '../16visualdata/dataset/'
        npzfile = np.load(path + 'data.npz')
        data = npzfile['data']
        data_arr = data
        label_arr_onehot = data
        return data_arr, label_arr_onehot
    else:
        return False