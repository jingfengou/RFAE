import numpy as np
import pandas as pd
import os
def write_to_csv(p_data, p_path):
    dataframe = pd.DataFrame(p_data)
    dataframe.to_csv(p_path, mode='a', header=False, index=False, sep=',')
    del dataframe
gene_dataset_path = ['15GEO']
# %%
fold_path_low = ['FAE', 'RFAE_dw']
k_feature_num = ['500', '600', '700', '800', '900', '943']

# %%
cv_times_from = 0
cv_times_to = 5
# %%
file_path = "/log/"

rec_low_collect = {}

for low_dimension_dataset_path_i in gene_dataset_path:
    for fold_path_low_i in fold_path_low:


        for k in k_feature_num:
            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + "_" + k + file_path + fold_path_low_i + k + '_results.csv'
            # print(path_results)
            if os.path.exists(path_results):
                results_analysis = np.array(pd.read_csv(path_results, header=None))

                results_analysis_linear_rec_ = results_analysis[-5:, 0][np.where(results_analysis[-5:, 0] >= 0)]
                cv_cal_times = len(results_analysis_linear_rec_)

                # cv_cal_times = len(results_analysis_test_acc_)
                if cv_cal_times >= (cv_times_to - cv_times_from):


                    results_analysis_linear_rec = results_analysis[-5:, 0][np.where(results_analysis[-5:, 0] >= 0)]

                    results_analysis_linear_rec = results_analysis_linear_rec[
                        np.where(results_analysis_linear_rec < 2000)]

                    rec_low_collect[k, fold_path_low_i] = [
                        "%.3f" % np.mean(results_analysis_linear_rec),
                        "%.3f" % np.std(results_analysis_linear_rec)]

                else:
                    rec_low_collect[k, fold_path_low_i] = [-100, -100]
            else:
                rec_low_collect[k, fold_path_low_i] = [-100, -100]

print(rec_low_collect)
item_index = 0
rec_collect = []

for low_dimension_dataset_path_i in gene_dataset_path:

    for k in k_feature_num:

        rec_collect_ = []
        for fold_path_low_i in fold_path_low:

            if (k, fold_path_low_i) in rec_low_collect.keys():
                rec_collect_.append(rec_low_collect[k, fold_path_low_i][item_index])

        rec_collect.append(np.array(rec_collect_).astype("float32"))

# %%
write_to_csv(rec_collect, str(cv_times_to) + 'rec_mean.csv')
print(rec_collect)
for i in rec_collect:
    print(len(i))

np.set_printoptions(suppress=True)
np.array(rec_collect)
print(np.array(rec_collect).shape)
for i in np.arange(np.array(rec_collect).shape[1]):
    print(np.sum(np.array(rec_collect)[:,i][np.where(np.array(rec_collect)[:,i]>0)])/np.array(rec_collect).shape[0])
item_index = 1  # 0 acc mean, std, rec mean, std
rec_collect = []

for low_dimension_dataset_path_i in gene_dataset_path:
    for k in k_feature_num:

        rec_collect_ = []
        for fold_path_low_i in fold_path_low:

            if (k, fold_path_low_i) in rec_low_collect.keys():
                rec_collect_.append(rec_low_collect[k, fold_path_low_i][item_index])

        rec_collect.append(np.array(rec_collect_).astype("float32"))

# %%
write_to_csv(rec_collect, str(cv_times_to) + 'rec_std.csv')
np.set_printoptions(suppress=True)
np.array(rec_collect)
for i in np.arange(np.array(rec_collect).shape[1]):
    print(np.sum(np.array(rec_collect)[:,i][np.where(np.array(rec_collect)[:,i]>0)])/np.array(rec_collect).shape[0])
