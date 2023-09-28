import numpy as np
import pandas as pd
import os
def write_to_csv(p_data, p_path):
    dataframe = pd.DataFrame(p_data)
    dataframe.to_csv(p_path, mode='a', header=False, index=False, sep=',')
    del dataframe
low_dimension_dataset_path = ['2COIL', '3Activity', '4ISOLET', '5MNIST', '6MNIST-Fashion', '7USPS']
low_mice_dimension_dataset_path = ['1MiceProtein']
high_dimension_dataset_path = ['8GLIOMA', '9leukemia', '10pixraw10P', '11ProstateGE', '12warpAR10P', '13SMKCAN187',
                               '14arcene']
# %%
fold_path_low = ['1_LS_CV', '2_SPEC_CV', '3_NDFS_CV', '4_AEFS_CV', '5_UDFS_CV', '6_MCFS_CV', '7_PFA_CV', '8_InfFS_CV',
                 '9_AgnoSS_CV', '10_CAE_CV', 'FAE', 'RFAE_exp_dw']

fold_path_low_mice = ['1_LS_CV', '2_SPEC_CV', '3_NDFS_CV', '4_AEFS_CV', '5_UDFS_CV', '6_MCFS_CV', '7_PFA_CV', '8_InfFS_CV',
                 '9_AgnoSS_CV', '10_CAE_CV', 'FAE', 'RFAE_exp_dw']

fold_path_high = ['1_LS_CV', '2_SPEC_CV', '3_NDFS_CV', '4_AEFS_CV', '5_UDFS_CV', '6_MCFS_CV', '7_PFA_CV', '8_InfFS_CV',
                 '9_AgnoSS_CV', '10_CAE_CV', 'FAE', 'RFAE_exp_dw']
# %%
cv_times_from = 0
cv_times_to = 5
# %%
file_path = "/log/"

acc_rec_low_collect = {}

for low_dimension_dataset_path_i in low_dimension_dataset_path:
    for fold_path_low_i in fold_path_low:

        if fold_path_low_i in ['1_LS_CV', '2_SPEC_CV', '3_NDFS_CV', '4_AEFS_CV', '5_UDFS_CV', '6_MCFS_CV', '7_PFA_CV',
                               '8_InfFS_CV', '9_AgnoSS_CV', '10_CAE_CV']:
            fold_low_name = fold_path_low_i.split('_')[1]
            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + file_path + fold_low_name + '_results.csv'
            # print(path_results)
            if os.path.exists(path_results):
                results_analysis = np.array(pd.read_csv(path_results, header=None))
                results_analysis_test_acc__ = results_analysis[:, 3]
                results_analysis_test_acc_ = results_analysis_test_acc__[np.where(results_analysis[:, 3] >= 0)]
                cv_cal_times = len(results_analysis_test_acc__)
                # cv_cal_times = len(results_analysis_test_acc_)
                if cv_cal_times >= (cv_times_to - cv_times_from):

                    results_analysis_test_acc = results_analysis_test_acc_[cv_times_from:cv_times_to]
                    results_analysis_linear_rec = results_analysis[:, 4][np.where(results_analysis[:, 4] >= 0)][
                                                  cv_times_from:cv_times_to]
                    results_analysis_linear_rec = results_analysis_linear_rec[
                        np.where(results_analysis_linear_rec < 2000)]
                    acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i] = [
                        "%.3f" % np.mean(results_analysis_test_acc),
                        "%.3f" % np.std(results_analysis_test_acc),
                        "%.3f" % np.mean(results_analysis_linear_rec),
                        "%.3f" % np.std(results_analysis_linear_rec)]
                else:
                    acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i] = [-100, -100, -100, -100]
            else:
                acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i] = [-100, -100, -100, -100]



        elif fold_path_low_i in ['FAE', 'RFAE_exp_dw']:
            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + '_50' + file_path + fold_path_low_i + '50' + '_results.csv'
            # print(path_results)
            if os.path.exists(path_results):

                results_analysis = np.array(pd.read_csv(path_results, header=None))
                results_analysis_test_acc__ = results_analysis[-5:, 3]


                results_analysis_test_acc_ = results_analysis_test_acc__[np.where(results_analysis[-5:, 3] >= 0)]
                cv_cal_times = len(results_analysis_test_acc__)

                # cv_cal_times = len(results_analysis_test_acc_)
                if cv_cal_times >= (cv_times_to - cv_times_from):

                    results_analysis_test_acc = results_analysis_test_acc_[:]

                    results_analysis_linear_rec = results_analysis[-5:, 4][np.where(results_analysis[-5:, 4] >= 0)]

                    results_analysis_linear_rec = results_analysis_linear_rec[
                        np.where(results_analysis_linear_rec < 2000)]

                    acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i+'50'] = [
                        "%.3f" % np.mean(results_analysis_test_acc),
                        "%.3f" % np.std(results_analysis_test_acc),
                        "%.3f" % np.mean(results_analysis_linear_rec),
                        "%.3f" % np.std(results_analysis_linear_rec)]

                else:
                    acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i+'50'] = [-100, -100, -100, -100]
            else:
                acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i+'50'] = [-100, -100, -100, -100]
            # print(acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i+'50'])

acc_rec_low_mice_collect = {}
print(acc_rec_low_collect)

for low_dimension_dataset_path_i in low_mice_dimension_dataset_path:
    for fold_path_low_i in fold_path_low_mice:

        if fold_path_low_i in ['1_LS_CV', '2_SPEC_CV', '3_NDFS_CV', '4_AEFS_CV', '5_UDFS_CV', '6_MCFS_CV', '7_PFA_CV',
                               '8_InfFS_CV', '9_AgnoSS_CV', '10_CAE_CV']:
            fold_low_name = fold_path_low_i.split('_')[1]
            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + file_path + fold_low_name + '_results.csv'
            # print(path_results)

            if os.path.exists(path_results):
                results_analysis = np.array(pd.read_csv(path_results, header=None))
                results_analysis_test_acc__ = results_analysis[:, 3]
                results_analysis_test_acc_ = results_analysis_test_acc__[np.where(results_analysis[:, 3] >= 0)]
                cv_cal_times = len(results_analysis_test_acc__)
                # cv_cal_times = len(results_analysis_test_acc_)
                if cv_cal_times >= (cv_times_to - cv_times_from):
                    results_analysis_test_acc = results_analysis_test_acc_[cv_times_from:cv_times_to]
                    results_analysis_linear_rec = results_analysis[:, 4][np.where(results_analysis[:, 4] >= 0)][
                                                  cv_times_from:cv_times_to]
                    results_analysis_linear_rec = results_analysis_linear_rec[
                        np.where(results_analysis_linear_rec < 2000)]
                    acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i] = [
                        "%.3f" % np.mean(results_analysis_test_acc),
                        "%.3f" % np.std(results_analysis_test_acc),
                        "%.3f" % np.mean(results_analysis_linear_rec),
                        "%.3f" % np.std(results_analysis_linear_rec)]
                else:
                    acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i] = [-100, -100, -100, -100]
            else:
                acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i] = [-100, -100, -100, -100]
        elif fold_path_low_i in ['FAE', 'RFAE_exp_dw']:

            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + '_10' + file_path + fold_path_low_i + '10' + '_results.csv'
            # print(path_results)

            if os.path.exists(path_results):
                results_analysis = np.array(pd.read_csv(path_results, header=None))
                results_analysis_test_acc__ = results_analysis[-5:, 3]


                results_analysis_test_acc_ = results_analysis_test_acc__[np.where(results_analysis[-5:, 3] >= 0)]
                cv_cal_times = len(results_analysis_test_acc__)

                # cv_cal_times = len(results_analysis_test_acc_)
                if cv_cal_times >= (cv_times_to - cv_times_from):

                    results_analysis_test_acc = results_analysis_test_acc_[:]

                    results_analysis_linear_rec = results_analysis[-5:, 4][np.where(results_analysis[-5:, 4] >= 0)]

                    results_analysis_linear_rec = results_analysis_linear_rec[
                        np.where(results_analysis_linear_rec < 2000)]

                    acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i+ '10'] = [
                        "%.3f" % np.mean(results_analysis_test_acc),
                        "%.3f" % np.std(results_analysis_test_acc),
                        "%.3f" % np.mean(results_analysis_linear_rec),
                        "%.3f" % np.std(results_analysis_linear_rec)]

                else:
                    acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i+ '10'] = [-100, -100, -100, -100]
            else:
                acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i+ '10'] = [-100, -100, -100, -100]

acc_rec_high_collect = {}

for low_dimension_dataset_path_i in high_dimension_dataset_path:
    for fold_path_low_i in fold_path_high:
        if fold_path_low_i in ['1_LS_CV', '2_SPEC_CV', '3_NDFS_CV', '4_AEFS_CV', '5_UDFS_CV', '6_MCFS_CV', '7_PFA_CV',
                               '8_InfFS_CV', '9_AgnoSS_CV', '10_CAE_CV']:
            fold_low_name = fold_path_low_i.split('_')[1]
            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + file_path + fold_low_name + '_results.csv'
            # print(path_results)

            if os.path.exists(path_results):
                results_analysis = np.array(pd.read_csv(path_results, header=None))
                results_analysis_test_acc__ = results_analysis[:, 3]
                results_analysis_test_acc_ = results_analysis_test_acc__[np.where(results_analysis[:, 3] >= 0)]
                cv_cal_times = len(results_analysis_test_acc__)
                # cv_cal_times = len(results_analysis_test_acc_)
                if cv_cal_times >= (cv_times_to - cv_times_from):
                    results_analysis_test_acc = results_analysis_test_acc_[cv_times_from:cv_times_to]
                    results_analysis_linear_rec = results_analysis[:, 4][np.where(results_analysis[:, 4] >= 0)][
                                                  cv_times_from:cv_times_to]
                    results_analysis_linear_rec = results_analysis_linear_rec[
                        np.where(results_analysis_linear_rec < 2000)]
                    acc_rec_high_collect[low_dimension_dataset_path_i, fold_path_low_i] = [
                        "%.3f" % np.mean(results_analysis_test_acc),
                        "%.3f" % np.std(results_analysis_test_acc),
                        "%.3f" % np.mean(results_analysis_linear_rec),
                        "%.3f" % np.std(results_analysis_linear_rec)]
                else:
                    acc_rec_high_collect[low_dimension_dataset_path_i, fold_path_low_i] = [-100, -100, -100, -100]
            else:
                acc_rec_high_collect[low_dimension_dataset_path_i, fold_path_low_i] = [-100, -100, -100, -100]
        elif fold_path_low_i in ['FAE', 'RFAE_exp_dw']:
            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + '_64' + file_path + fold_path_low_i + '64' + '_results.csv'
            # print(path_results)
            if os.path.exists(path_results):
                results_analysis = np.array(pd.read_csv(path_results, header=None))
                results_analysis_test_acc__ = results_analysis[-5:, 3]


                results_analysis_test_acc_ = results_analysis_test_acc__[np.where(results_analysis[-5:, 3] >= 0)]
                cv_cal_times = len(results_analysis_test_acc__)

                # cv_cal_times = len(results_analysis_test_acc_)
                if cv_cal_times >= (cv_times_to - cv_times_from):

                    results_analysis_test_acc = results_analysis_test_acc_[:]

                    results_analysis_linear_rec = results_analysis[-5:, 4][np.where(results_analysis[-5:, 4] >= 0)]

                    results_analysis_linear_rec = results_analysis_linear_rec[
                        np.where(results_analysis_linear_rec < 2000)]

                    acc_rec_high_collect[low_dimension_dataset_path_i, fold_path_low_i+ '64'] = [
                        "%.3f" % np.mean(results_analysis_test_acc),
                        "%.3f" % np.std(results_analysis_test_acc),
                        "%.3f" % np.mean(results_analysis_linear_rec),
                        "%.3f" % np.std(results_analysis_linear_rec)]

                else:
                    acc_rec_high_collect[low_dimension_dataset_path_i, fold_path_low_i+ '64'] = [-100, -100, -100, -100]
            else:
                acc_rec_high_collect[low_dimension_dataset_path_i, fold_path_low_i+ '64'] = [-100, -100, -100, -100]

# item_index = 0  # 0 acc mean, std, rec mean, std
# acc_collect = []
#
# for low_dimension_dataset_path_i in low_mice_dimension_dataset_path:
#     acc_collect_ = []
#     for fold_path_low_i in fold_path_low_mice:
#         if (low_dimension_dataset_path_i, fold_path_low_i) in acc_rec_low_mice_collect.keys():
#             acc_collect_.append(acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i][item_index])
#         elif (low_dimension_dataset_path_i, fold_path_low_i + '10') in acc_rec_low_mice_collect.keys():
#             acc_collect_.append(acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i + '10'][item_index])
#
#
#     acc_collect.append(np.array(acc_collect_).astype("float32"))
#
# for low_dimension_dataset_path_i in low_dimension_dataset_path:
#     acc_collect_ = []
#     for fold_path_low_i in fold_path_low:
#         if (low_dimension_dataset_path_i, fold_path_low_i) in acc_rec_low_collect.keys():
#             acc_collect_.append(acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i][item_index])
#         elif (low_dimension_dataset_path_i, fold_path_low_i + '50') in acc_rec_low_collect.keys():
#             acc_collect_.append(acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i + '50'][item_index])
#
#
#     acc_collect.append(np.array(acc_collect_).astype("float32"))
#
# for high_dimension_dataset_path_i in high_dimension_dataset_path:
#     acc_collect_ = []
#     for fold_path_high_i in fold_path_high:
#         if (high_dimension_dataset_path_i, fold_path_high_i) in acc_rec_high_collect.keys():
#             acc_collect_.append(acc_rec_high_collect[high_dimension_dataset_path_i, fold_path_high_i][item_index])
#         elif (high_dimension_dataset_path_i, fold_path_high_i + '64') in acc_rec_high_collect.keys():
#             acc_collect_.append(acc_rec_high_collect[high_dimension_dataset_path_i, fold_path_high_i+ '64'][item_index])
#
#
#     acc_collect.append(np.array(acc_collect_).astype("float32"))
# # %%
# write_to_csv(acc_collect, str(cv_times_to) + 'acc_mean.csv')
# print(acc_collect)
# for i in acc_collect:
#     print(len(i))
#
# np.set_printoptions(suppress=True)
# np.array(acc_collect)
# print(np.array(acc_collect).shape)
# for i in np.arange(np.array(acc_collect).shape[1]):
#     print(np.sum(np.array(acc_collect)[:,i][np.where(np.array(acc_collect)[:,i]>0)])/np.array(acc_collect).shape[0])
# item_index = 1  # 0 acc mean, std, rec mean, std
# acc_collect = []
#
# for low_dimension_dataset_path_i in low_mice_dimension_dataset_path:
#     acc_collect_ = []
#     for fold_path_low_i in fold_path_low_mice:
#         if (low_dimension_dataset_path_i, fold_path_low_i) in acc_rec_low_mice_collect.keys():
#             acc_collect_.append(acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i][item_index])
#         elif (low_dimension_dataset_path_i, fold_path_low_i + '10') in acc_rec_low_mice_collect.keys():
#             acc_collect_.append(acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i+'10'][item_index])
#
#
#     acc_collect.append(np.array(acc_collect_).astype("float32"))
#
# for low_dimension_dataset_path_i in low_dimension_dataset_path:
#     acc_collect_ = []
#     for fold_path_low_i in fold_path_low:
#         if (low_dimension_dataset_path_i, fold_path_low_i) in acc_rec_low_collect.keys():
#             acc_collect_.append(acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i][item_index])
#         elif (low_dimension_dataset_path_i, fold_path_low_i + '50') in acc_rec_low_collect.keys():
#             acc_collect_.append(acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i+'50'][item_index])
#
#
#
#     acc_collect.append(np.array(acc_collect_).astype("float32"))
#
# for high_dimension_dataset_path_i in high_dimension_dataset_path:
#     acc_collect_ = []
#     for fold_path_high_i in fold_path_high:
#         if (high_dimension_dataset_path_i, fold_path_high_i) in acc_rec_high_collect.keys():
#             acc_collect_.append(acc_rec_high_collect[high_dimension_dataset_path_i, fold_path_high_i][item_index])
#         elif (high_dimension_dataset_path_i, fold_path_high_i + '64') in acc_rec_high_collect.keys():
#             acc_collect_.append(acc_rec_high_collect[high_dimension_dataset_path_i, fold_path_high_i+'64'][item_index])
#
#
#
#     acc_collect.append(np.array(acc_collect_).astype("float32"))
# # %%
#
# write_to_csv(acc_collect, str(cv_times_to) + 'acc_std.csv')
# np.set_printoptions(suppress=True)
# np.array(acc_collect)
# for i in np.arange(np.array(acc_collect).shape[1]):
#     print(np.sum(np.array(acc_collect)[:,i][np.where(np.array(acc_collect)[:,i]>0)])/np.array(acc_collect).shape[0])
item_index = 2  # 0 acc mean, 1 std, 2 rec mean, 3 std
acc_collect = []
for low_dimension_dataset_path_i in low_mice_dimension_dataset_path:
    acc_collect_ = []
    for fold_path_low_i in fold_path_low_mice:
        if (low_dimension_dataset_path_i, fold_path_low_i) in acc_rec_low_mice_collect.keys():
            acc_collect_.append(acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i][item_index])
        elif (low_dimension_dataset_path_i, fold_path_low_i + '10') in acc_rec_low_mice_collect.keys():
            acc_collect_.append(acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i+'10'][item_index])
            # acc_collect_.append(acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i + '8'][item_index])

    acc_collect.append(np.array(acc_collect_).astype("float32"))

for low_dimension_dataset_path_i in low_dimension_dataset_path:
    acc_collect_ = []
    for fold_path_low_i in fold_path_low:
        if (low_dimension_dataset_path_i, fold_path_low_i) in acc_rec_low_collect.keys():
            acc_collect_.append(acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i][item_index])
        elif (low_dimension_dataset_path_i, fold_path_low_i + '50') in acc_rec_low_collect.keys():
            acc_collect_.append(acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i+'50'][item_index])
            # acc_collect_.append(acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i + '36'][item_index])


    acc_collect.append(np.array(acc_collect_).astype("float32"))

for high_dimension_dataset_path_i in high_dimension_dataset_path:
    acc_collect_ = []
    for fold_path_high_i in fold_path_high:
        if (high_dimension_dataset_path_i, fold_path_high_i) in acc_rec_high_collect.keys():
            acc_collect_.append(acc_rec_high_collect[high_dimension_dataset_path_i, fold_path_high_i][item_index])
        elif (high_dimension_dataset_path_i, fold_path_high_i + '64') in acc_rec_high_collect.keys():
            acc_collect_.append(acc_rec_high_collect[high_dimension_dataset_path_i, fold_path_high_i+'64'][item_index])
            # acc_collect_.append(acc_rec_high_collect[high_dimension_dataset_path_i, fold_path_high_i + '50'][item_index])


    acc_collect.append(np.array(acc_collect_).astype("float32"))

write_to_csv(acc_collect, str(cv_times_to) + 'rec_mean.csv')
np.set_printoptions(suppress=True)
np.array(acc_collect)
item_index = 3  # 0 acc mean, 1 std, 2 rec mean, 3 std
acc_collect = []
for low_dimension_dataset_path_i in low_mice_dimension_dataset_path:
    acc_collect_ = []
    for fold_path_low_i in fold_path_low_mice:
        if (low_dimension_dataset_path_i, fold_path_low_i) in acc_rec_low_mice_collect.keys():
            acc_collect_.append(acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i][item_index])
        elif (low_dimension_dataset_path_i, fold_path_low_i + '10') in acc_rec_low_mice_collect.keys():
            acc_collect_.append(acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i+'10'][item_index])
            # acc_collect_.append(acc_rec_low_mice_collect[low_dimension_dataset_path_i, fold_path_low_i + '8'][item_index])

    acc_collect.append(np.array(acc_collect_).astype("float32"))

for low_dimension_dataset_path_i in low_dimension_dataset_path:
    acc_collect_ = []
    for fold_path_low_i in fold_path_low:
        if (low_dimension_dataset_path_i, fold_path_low_i) in acc_rec_low_collect.keys():
            acc_collect_.append(acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i][item_index])
        elif (low_dimension_dataset_path_i, fold_path_low_i + '50') in acc_rec_low_collect.keys():
            acc_collect_.append(acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i+'50'][item_index])
            # acc_collect_.append(acc_rec_low_collect[low_dimension_dataset_path_i, fold_path_low_i + '36'][item_index])


    acc_collect.append(np.array(acc_collect_).astype("float32"))

for high_dimension_dataset_path_i in high_dimension_dataset_path:
    acc_collect_ = []
    for fold_path_high_i in fold_path_high:
        if (high_dimension_dataset_path_i, fold_path_high_i) in acc_rec_high_collect.keys():
            acc_collect_.append(acc_rec_high_collect[high_dimension_dataset_path_i, fold_path_high_i][item_index])
        elif (high_dimension_dataset_path_i, fold_path_high_i + '64') in acc_rec_high_collect.keys():
            acc_collect_.append(acc_rec_high_collect[high_dimension_dataset_path_i, fold_path_high_i+'64'][item_index])
            # acc_collect_.append(acc_rec_high_collect[high_dimension_dataset_path_i, fold_path_high_i + '50'][item_index])


    acc_collect.append(np.array(acc_collect_).astype("float32"))
# %%
write_to_csv(acc_collect, str(cv_times_to) + 'rec_std.csv')
np.set_printoptions(suppress=True)
np.array(acc_collect)