import numpy as np
import pandas as pd
import os
import csv
def write_to_csv(p_data, p_path):
    dataframe = pd.DataFrame(p_data)
    dataframe.to_csv(p_path, mode='a', header=False, index=False, sep=',')
    del dataframe
low_dimension_dataset_path = ['2COIL', '3Activity', '4ISOLET', '5MNIST', '6MNIST-Fashion', '7USPS']
low_mice_dimension_dataset_path = ['1MiceProtein']
high_dimension_dataset_path = ['8GLIOMA', '9leukemia', '10pixraw10P', '11ProstateGE', '12warpAR10P', '13SMKCAN187',
                               '14arcene']
gene_dataset_path = ['15GEO']
# %%
fold_path_low = ['FAE', 'RFAE_exp', 'RFAE_dw', 'RFAE_exp_dw']

fold_path_low_mice = ['FAE', 'RFAE_exp', 'RFAE_dw', 'RFAE_exp_dw']

fold_path_high = ['FAE', 'RFAE_exp', 'RFAE_dw', 'RFAE_exp_dw']
# %%
cv_times_from = 0
cv_times_to = 5
# %%
file_path = "/log/"

overlap_collect = []
overlap_low_collect = []
featurenum_low_collect=[]
for low_dimension_dataset_path_i in low_dimension_dataset_path:
    for fold_path_low_i in fold_path_low:
        if fold_path_low_i in ['FAE', 'RFAE_exp', 'RFAE_dw', 'RFAE_exp_dw']:

            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + '_50' + file_path + fold_path_low_i + '50' + "_selected_list" + ".csv"

            if os.path.exists(path_results):
                results_analysis = []
                with open(path_results, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        results_analysis.append(row)

                cv_cal_times = len(results_analysis)
                # print("cv times: ",cv_cal_times)
                if cv_cal_times >= (cv_times_to - cv_times_from):
                    sum = 0
                    cnt = 0
                    k = 0
                    for i in range(cv_cal_times - 5, cv_cal_times):
                        for j in range(i, cv_cal_times):
                            set1 = set(results_analysis[i])
                            set2 = set(results_analysis[j])

                            interset = set2.intersection(set1)
                            sum += + len(interset)
                            cnt += 1
                        # print(list(set1))
                        k += len(results_analysis[i])
                    mean = sum / cnt
                    k /= 5
                    featurenum_low_collect.append(k)
                    # print(mean)
                    overlap_collect.append(mean)

        overlap_low_collect.append(overlap_collect)
        overlap_collect = []

overlap_low_collect = np.array(overlap_low_collect)
featurenum_low_collect = np.array(featurenum_low_collect)
overlap_low_collect = overlap_low_collect.reshape( len(overlap_low_collect) // (len(fold_path_low)*1), len(fold_path_low)*1)
featurenum_low_collect = featurenum_low_collect.reshape(len(featurenum_low_collect) // (len(fold_path_low)*1), len(fold_path_low)*1)
print(overlap_low_collect.shape)
# print(overlap_collect)
overlap_low_mice_collect_ = []
overlap_low_mice_collect = []
featurenum_low_mice_collect = []
for low_dimension_dataset_path_i in low_mice_dimension_dataset_path:
    for fold_path_low_i in fold_path_low_mice:

        # fold_low_name = fold_path_low_i.split('_')[1]

        if fold_path_low_i in ['FAE', 'RFAE_exp', 'RFAE_dw', 'RFAE_exp_dw']:

            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + '_10' + file_path + fold_path_low_i + '10' + "_selected_list" + ".csv"

            if os.path.exists(path_results):
                results_analysis = []
                with open(path_results, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        results_analysis.append(row)

                cv_cal_times = len(results_analysis)
                # print("cv times: ",cv_cal_times)
                if cv_cal_times >= (cv_times_to - cv_times_from):
                    sum = 0
                    cnt = 0
                    k = 0
                    for i in range(cv_cal_times - 5, cv_cal_times):
                        for j in range(i, cv_cal_times):
                            set1 = set(results_analysis[i])
                            set2 = set(results_analysis[j])

                            interset = set2.intersection(set1)
                            sum += + len(interset)
                            cnt += 1
                        # print(list(set1))
                        k += len(results_analysis[i])
                    mean = sum / cnt
                    k /= 5
                    featurenum_low_mice_collect.append(k)

                    # print(mean)
                    overlap_low_mice_collect.append(mean)

        overlap_low_mice_collect_.append(overlap_low_mice_collect)

        overlap_low_mice_collect = []

overlap_low_mice_collect_ = np.array(overlap_low_mice_collect_)

overlap_low_mice_collect_ = overlap_low_mice_collect_.reshape(len(overlap_low_mice_collect_) // (len(fold_path_low_mice)*1), len(fold_path_low_mice)*1)
featurenum_low_mice_collect = np.array(featurenum_low_mice_collect)
featurenum_low_mice_collect = featurenum_low_mice_collect.reshape(len(featurenum_low_mice_collect) // (len(fold_path_low_mice)*1), len(fold_path_low_mice)*1)
print(overlap_low_mice_collect_.shape)
overlap_high_collect = []
overlap_high_collect_ = []
featurenum_high_collect = []
for low_dimension_dataset_path_i in high_dimension_dataset_path:
    for fold_path_low_i in fold_path_high:
        # fold_low_name = fold_path_low_i.split('_')[1]




        if fold_path_low_i in ['FAE', 'RFAE_exp', 'RFAE_dw', 'RFAE_exp_dw']:

            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + '_64' + file_path + fold_path_low_i + '64' + "_selected_list" + ".csv"


            if os.path.exists(path_results):
                results_analysis = []
                with open(path_results, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        results_analysis.append(row)
                cv_cal_times = len(results_analysis)
                # print("cv times: ",cv_cal_times)
                if cv_cal_times >= (cv_times_to - cv_times_from):
                    sum = 0
                    cnt = 0
                    k = 0
                    for i in range(cv_cal_times - 5, cv_cal_times):
                        for j in range(i, cv_cal_times):
                            set1 = set(results_analysis[i])
                            set2 = set(results_analysis[j])

                            interset = set2.intersection(set1)
                            sum += + len(interset)
                            cnt += 1
                        # print(list(set1))
                        k += len(results_analysis[i])
                    mean = sum / cnt
                    k /= 5
                    featurenum_high_collect.append(k)

                    # print(mean)
                    overlap_high_collect.append(mean)



        overlap_high_collect_.append(overlap_high_collect)

        overlap_high_collect = []
overlap_high_collect_ = np.array(overlap_high_collect_)

overlap_high_collect_ = overlap_high_collect_.reshape(len(overlap_high_collect_) // (len(fold_path_high)*1), len(fold_path_high)*1)

featurenum_high_collect = np.array(featurenum_high_collect)
featurenum_high_collect = featurenum_high_collect.reshape(len(featurenum_high_collect) // (len(fold_path_high)*1), len(fold_path_high)*1)

print(overlap_high_collect_.shape)
# print(overlap_low_collect)
# print(overlap_low_mice_collect_)
# print(overlap_high_collect_)

overlap_gene_collect = []
overlap_gene_collect_ = []
featurenum_gene_collect = []
for low_dimension_dataset_path_i in gene_dataset_path:
    for fold_path_low_i in fold_path_high:
        # fold_low_name = fold_path_low_i.split('_')[1]

        if fold_path_low_i in ['FAE', 'RFAE_exp', 'RFAE_dw', 'RFAE_exp_dw']:

            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + '_500' + file_path + fold_path_low_i + '500' + "_selected_list" + ".csv"


            if os.path.exists(path_results):
                results_analysis = []
                with open(path_results, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        results_analysis.append(row)
                cv_cal_times = len(results_analysis)
                # print("cv times: ",cv_cal_times)
                if cv_cal_times >= (cv_times_to - cv_times_from):
                    sum = 0
                    cnt = 0
                    k = 0
                    for i in range(cv_cal_times - 5, cv_cal_times):
                        for j in range(i, cv_cal_times):
                            set1 = set(results_analysis[i])
                            set2 = set(results_analysis[j])

                            interset = set2.intersection(set1)
                            sum += + len(interset)
                            cnt += 1
                        # print(list(set1))
                        k += len(results_analysis[i])
                    mean = sum / cnt
                    k /= 5
                    featurenum_gene_collect.append(k)

                    # print(mean)
                    overlap_gene_collect.append(mean)
            overlap_gene_collect_.append(overlap_gene_collect)

            overlap_gene_collect = []
            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + '_600' + file_path + fold_path_low_i + '600' + "_selected_list" + ".csv"


            if os.path.exists(path_results):
                results_analysis = []
                with open(path_results, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        results_analysis.append(row)
                cv_cal_times = len(results_analysis)
                # print("cv times: ",cv_cal_times)
                if cv_cal_times >= (cv_times_to - cv_times_from):
                    sum = 0
                    cnt = 0
                    k = 0
                    for i in range(cv_cal_times - 5, cv_cal_times):
                        for j in range(i, cv_cal_times):
                            set1 = set(results_analysis[i])
                            set2 = set(results_analysis[j])

                            interset = set2.intersection(set1)
                            sum += + len(interset)
                            cnt += 1
                        # print(list(set1))
                        k += len(results_analysis[i])
                    mean = sum / cnt
                    k /= 5
                    featurenum_gene_collect.append(k)

                    # print(mean)
                    overlap_gene_collect.append(mean)
            overlap_gene_collect_.append(overlap_gene_collect)
            overlap_gene_collect = []
            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + '_700' + file_path + fold_path_low_i + '700' + "_selected_list" + ".csv"


            if os.path.exists(path_results):
                results_analysis = []
                with open(path_results, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        results_analysis.append(row)
                cv_cal_times = len(results_analysis)
                # print("cv times: ",cv_cal_times)
                if cv_cal_times >= (cv_times_to - cv_times_from):
                    sum = 0
                    cnt = 0
                    k = 0
                    for i in range(cv_cal_times - 5, cv_cal_times):
                        for j in range(i, cv_cal_times):
                            set1 = set(results_analysis[i])
                            set2 = set(results_analysis[j])

                            interset = set2.intersection(set1)
                            sum += + len(interset)
                            cnt += 1
                        # print(list(set1))
                        k += len(results_analysis[i])
                    mean = sum / cnt
                    k /= 5
                    featurenum_gene_collect.append(k)

                    # print(mean)
                    overlap_gene_collect.append(mean)
            overlap_gene_collect_.append(overlap_gene_collect)
            overlap_gene_collect = []
            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + '_800' + file_path + fold_path_low_i + '800' + "_selected_list" + ".csv"


            if os.path.exists(path_results):
                results_analysis = []
                with open(path_results, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        results_analysis.append(row)
                cv_cal_times = len(results_analysis)
                # print("cv times: ",cv_cal_times)
                if cv_cal_times >= (cv_times_to - cv_times_from):
                    sum = 0
                    cnt = 0
                    k = 0
                    for i in range(cv_cal_times - 5, cv_cal_times):
                        for j in range(i, cv_cal_times):
                            set1 = set(results_analysis[i])
                            set2 = set(results_analysis[j])

                            interset = set2.intersection(set1)
                            sum += + len(interset)
                            cnt += 1
                        # print(list(set1))
                        k += len(results_analysis[i])
                    mean = sum / cnt
                    k /= 5
                    featurenum_gene_collect.append(k)

                    # print(mean)
                    overlap_gene_collect.append(mean)
            overlap_gene_collect_.append(overlap_gene_collect)
            overlap_gene_collect = []
            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + '_900' + file_path + fold_path_low_i + '900' + "_selected_list" + ".csv"


            if os.path.exists(path_results):
                results_analysis = []
                with open(path_results, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        results_analysis.append(row)
                cv_cal_times = len(results_analysis)
                # print("cv times: ",cv_cal_times)
                if cv_cal_times >= (cv_times_to - cv_times_from):
                    sum = 0
                    cnt = 0
                    k = 0
                    for i in range(cv_cal_times - 5, cv_cal_times):
                        for j in range(i, cv_cal_times):
                            set1 = set(results_analysis[i])
                            set2 = set(results_analysis[j])

                            interset = set2.intersection(set1)
                            sum += + len(interset)
                            cnt += 1
                        # print(list(set1))
                        k += len(results_analysis[i])
                    mean = sum / cnt
                    k /= 5
                    featurenum_gene_collect.append(k)

                    # print(mean)
                    overlap_gene_collect.append(mean)
            overlap_gene_collect_.append(overlap_gene_collect)
            overlap_gene_collect = []
            path_results = '../../' + low_dimension_dataset_path_i + '/' + fold_path_low_i + '_943' + file_path + fold_path_low_i + '943' + "_selected_list" + ".csv"


            if os.path.exists(path_results):
                results_analysis = []
                with open(path_results, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        results_analysis.append(row)
                cv_cal_times = len(results_analysis)
                # print("cv times: ",cv_cal_times)
                if cv_cal_times >= (cv_times_to - cv_times_from):
                    sum = 0
                    cnt = 0
                    k = 0
                    for i in range(cv_cal_times - 5, cv_cal_times):
                        for j in range(i, cv_cal_times):
                            set1 = set(results_analysis[i])
                            set2 = set(results_analysis[j])

                            interset = set2.intersection(set1)
                            sum += + len(interset)
                            cnt += 1
                        # print(list(set1))
                        k += len(results_analysis[i])
                    mean = sum / cnt
                    k /= 5
                    featurenum_gene_collect.append(k)

                    # print(mean)
                    overlap_gene_collect.append(mean)
            overlap_gene_collect_.append(overlap_gene_collect)

            overlap_gene_collect = []
overlap_gene_collect_ = np.array(overlap_gene_collect_)
print(overlap_gene_collect_.shape)
overlap_gene_collect_ = overlap_gene_collect_.reshape(len(overlap_gene_collect_) // (len(fold_path_high)), len(fold_path_high))
print(overlap_gene_collect_.shape)
print(overlap_gene_collect_)
featurenum_gene_collect = np.array(featurenum_gene_collect)
featurenum_gene_collect = featurenum_gene_collect.reshape(len(featurenum_gene_collect) // (len(fold_path_high)*6), len(fold_path_high)*6)


overlap_result = np.concatenate((overlap_low_mice_collect_, overlap_low_collect, overlap_high_collect_), axis=0)

featurenum_result = np.concatenate((featurenum_low_mice_collect, featurenum_low_collect, featurenum_high_collect), axis=0)
print(overlap_result.shape)



write_to_csv(overlap_result, 'overlap_mean.csv')


write_to_csv(overlap_gene_collect_, 'gene_overlap_mean.csv')
write_to_csv(featurenum_result, 'featurenum_mean.csv')
