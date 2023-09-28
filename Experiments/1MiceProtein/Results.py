import numpy as np
import pandas as pd
import os

fold_path=['1_LS_CV','2_SPEC_CV','3_NDFS_CV','4_AEFS_CV','5_UDFS_CV','6_MCFS_CV',
           '7_PFA_CV','8_InfFS_CV','9_AgnoSS_CV','10_CAE_CV','11_FAE_10_CV','12_FAE_8_CV'
           , '13_FAEexp10_CV', '14_TFAE64_CV']

file_path = "/log/"
# for i in range(12, 14):
#     fold_name = fold_path[i].split('_')[1]
#     path_results = './' + fold_path[i] + file_path + fold_name + '_results.csv'
#     results_analysis = np.array(pd.read_csv(path_results, header=None))
#     print(results_analysis.shape)
#     if results_analysis.shape[1] > 50:
#         results_analysis = results_analysis[-50, :]
#     print(results_analysis.shape)


cv_times_from=0
cv_times_to=5


for fold_path_i in fold_path:
    fold_name = fold_path_i.split('_')[1]
    print(fold_name + ":\n")

    if fold_path_i in ['11_FAE_10_CV', '12_FAE_8_CV']:
        print("Do not use bias!", fold_path_i)
        p_is_use_bias = False
        path_results = './' + fold_path_i + file_path + "FAE_results_bias_" + str(p_is_use_bias) + ".csv"
        results_analysis = np.array(pd.read_csv(path_results, header=None))
        results_analysis_test_acc__ = results_analysis[:, 3]
        results_analysis_test_acc_ = results_analysis_test_acc__[np.where(results_analysis[:, 3] >= 0)]
        cv_cal_times = len(results_analysis_test_acc__)
        if cv_cal_times >= (cv_times_to - cv_times_from):
            print("cv times: ", cv_cal_times)

            results_analysis_test_acc = results_analysis_test_acc_[cv_times_from:cv_times_to]
            results_analysis_linear_rec = results_analysis[:, 4][np.where(results_analysis[:, 4] >= 0)][
                                          cv_times_from:cv_times_to]

            print("Testing accuarcy:%.3f" % np.mean(results_analysis_test_acc),
                  "±%.4f" % np.std(results_analysis_test_acc))
            print("Testing Linear reconstruction: %.3f" % np.mean(results_analysis_linear_rec),
                  "±%.4f" % np.std(results_analysis_linear_rec))

            path_time = './' + fold_path_i + file_path + "FAE_time_bias_" + str(p_is_use_bias) + ".csv"
            time_analysis = np.array(pd.read_csv(path_time, header=None))[cv_times_from:cv_times_to]

            print("Calculation times: ", len(results_analysis_test_acc))

            print("Use bias!", fold_path_i)
            p_is_use_bias = True
            path_results = './' + fold_path_i + file_path + "FAE_results_bias_" + str(p_is_use_bias) + ".csv"
            results_analysis = np.array(pd.read_csv(path_results, header=None))
            results_analysis_test_acc = results_analysis[:, 3][np.where(results_analysis[:, 3] >= 0)][
                                        cv_times_from:cv_times_to]
            results_analysis_linear_rec = results_analysis[:, 4][np.where(results_analysis[:, 4] >= 0)][
                                          cv_times_from:cv_times_to]

            print(results_analysis_test_acc)
            print(results_analysis_linear_rec)
            print("Testing accuarcy:%.3f" % np.mean(results_analysis_test_acc),
                  "±%.4f" % np.std(results_analysis_test_acc))
            print("Testing Linear reconstruction: %.3f" % np.mean(results_analysis_linear_rec),
                  "±%.4f" % np.std(results_analysis_linear_rec))

            path_time = './' + fold_path_i + file_path + "FAE_time_bias_" + str(p_is_use_bias) + ".csv"
            time_analysis = np.array(pd.read_csv(path_time, header=None))[cv_times_from:cv_times_to]

            print("Calculation times: ", len(results_analysis_test_acc))

    else:
        path_results = './' + fold_path_i + file_path + fold_name + '_results.csv'

        if os.path.exists(path_results):
            results_analysis = np.array(pd.read_csv(path_results, header=None))

            results_analysis_test_acc__ = results_analysis[:, 3]
            results_analysis_test_acc_ = results_analysis_test_acc__[np.where(results_analysis[:, 3] >= 0)]
            cv_cal_times = len(results_analysis_test_acc__)

            cv_cal_times = len(results_analysis_test_acc_)
            if cv_cal_times >= (cv_times_to - cv_times_from):
                print("cv times: ", cv_cal_times)

                results_analysis_test_acc = results_analysis_test_acc_[cv_times_from:cv_times_to]
                results_analysis_linear_rec = results_analysis[:, 4][np.where(results_analysis[:, 4] >= 0)][
                                              cv_times_from:cv_times_to]

                print("Testing accuarcy:%.3f" % np.mean(results_analysis_test_acc),
                      "±%.4f" % np.std(results_analysis_test_acc))
                print("Testing Linear reconstruction: %.3f" % np.mean(results_analysis_linear_rec),
                      "±%.4f" % np.std(results_analysis_linear_rec))

                path_time = './' + fold_path_i + file_path + fold_path_i.split('_')[1] + '_time.csv'
                time_analysis = np.array(pd.read_csv(path_time, header=None))[cv_times_from:cv_times_to]

                print("Calculation times: ", len(results_analysis_test_acc))

                print("\n\n")
        else:
            print("Nan")
            print("\n\n")