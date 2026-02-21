"""
================================================================================
CONTROL CHART MONITORING (PLMMD VERSION)
================================================================================
This script demonstrates the Phase I and Phase II monitoring process using
the PLMMD method.

NOTE FOR OTHER METHODS:
To adapt this script for benchmark methods (VPCA, Tucker, FPCA, etc.),
modify the 'Score Projection' section:
================================================================================
"""

import numpy as np
import pandas as pd
from data_generation import data_generation, oc_data_generation, oc2_data_generation, oc3_data_generation


def angle_between_vectors(a1, a2):
    dot_product = np.dot(a1, a2)
    norm_1 = np.linalg.norm(a1)
    norm_2 = np.linalg.norm(a2)
    cos_theta = dot_product / (norm_1 * norm_2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)


def find_adjusted_CL(CL_lower, CL_upper):
    while True:
        CL_mid = (CL_upper + CL_lower) / 2
        RL_list = []
        for seed in seeds_list:
            # phase 2
            n2 = 1000
            # get new data in control
            data_dict_2 = oc_data_generation(0, n2, seed * 3)
            scores_dict_2 = {}
            for key in keys:
                scores_dict_2[key] = data_dict_2[key] @ eigenvectors_dict_1[key]

            for key, matrix in scores_dict_2.items():
                scores_dict_2[key] = matrix / l_dict[key]

            all_matrices_2 = list(scores_dict_2.values())
            scores_2 = sum(all_matrices_2) / 5

            # Phase II T2 control chart
            T2_values_2 = []
            for i in range(scores_2.shape[0]):
                x = scores_2[i]
                diff = x - M
                T2 = np.dot(np.dot(diff.T, np.linalg.inv(S)), diff)
                T2_values_2.append(T2)
            T2_values_2 = np.array(T2_values_2)

            outlier_found = False
            for i, T2_value in enumerate(T2_values_2):
                if T2_value > CL_mid:
                    RL = i + 1
                    RL_list.append(RL)
                    # print(f"First outlier found at index: {RL}")
                    outlier_found = True
                    break
            # if not outlier_found:
            # print("No outliers found")

        ARL0 = np.mean(RL_list)
        print(ARL0)
        tolerance = np.abs(ARL0 - 100)
        if tolerance < 2.5:
            break

        if ARL0 < 100:
            CL_lower = CL_mid
        else:
            CL_upper = CL_mid

    return CL_mid


path = ""
keys = ['data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW']
pc_numbers = [7, 7, 6, 6, 8]
best_params = [364986, 555817, 511958, 550696, 612431]

seeds_list = [1501, 2586, 2653, 1055, 705, 106, 589, 2468, 2413, 1600,
              2464, 228, 915, 794, 3021, 3543, 1073, 3351, 1744, 1084,
              926, 3049, 1117, 642, 4767, 501, 4066, 333, 4684, 486,
              1962, 393, 4842, 4866, 1755, 2515, 3585, 4315, 4966, 2099,
              3599, 4121, 29, 65, 838, 3906, 3773, 4635, 3161, 2659,
              4615, 4628, 2451, 2846, 1144, 3078, 1103, 168, 1670, 2570,
              2377, 4395, 4257, 3862, 23, 2633, 3340, 2215, 3682, 4724,
              1907, 84, 227, 296, 1001, 2138, 711, 2801, 2527, 3752,
              3321, 3181, 4183, 1615, 3024, 1413, 763, 655, 4797, 3849,  # np.random.seed(42)
              4153, 468, 157, 1295, 497, 4740, 2940, 3456, 373, 79]

# Phase I
# get eigenvectors
n1 = 300
seed = 1001
data_dict_1 = data_generation(n1, seed)
eigenvectors_dict_1 = {}
scores_dict_1 = {}
for key in keys:
    b = pd.read_csv(f"/n={n1}, seed={seed}/{key}.csv")
    b = b.values
    v, _ = np.linalg.qr(b)
    eigenvectors_dict_1[key] = v

# get scores
for key in keys:
    scores_dict_1[key] = data_dict_1[key] @ eigenvectors_dict_1[key]

# scale
for key, matrix in scores_dict_1.items():
    norms = np.linalg.norm(matrix, axis=0)
    scores_dict_1[key] = matrix / norms

cluster_indices = pd.read_csv(path + "structure_5stations.csv")  # get structure
cluster_indices['indices'] = cluster_indices['indices'].str.strip('[]')
for idx, row in cluster_indices['indices'].items():
    indices_list = row.replace("'", "").split(", ")
    first_matrix_index, first_col_index = indices_list[0].split('_')
    first_matrix_index = int(first_matrix_index) - 1
    first_col_index = int(first_col_index) - 1
    first_key = keys[first_matrix_index]
    s1 = scores_dict_1[first_key][:, first_col_index]
    for num, indices in enumerate(indices_list[1:], start=1):
        matrix_index, col_index = indices.split('_')
        matrix_index = int(matrix_index) - 1
        col_index = int(col_index) - 1
        key = keys[matrix_index]
        s2 = scores_dict_1[key][:, col_index]
        angle = angle_between_vectors(s1, s2)
        if angle > 90:
            print(f'angle:{angle}')
            eigenvectors_dict_1[key][:, col_index] *= -1

scores_dict_2 = {}
for key in keys:
    scores_dict_2[key] = data_dict_1[key] @ eigenvectors_dict_1[key]

l_dict = {}
for key, matrix in scores_dict_2.items():
    norms = np.linalg.norm(matrix, axis=0)
    scores_dict_2[key] = matrix / norms
    l_dict[key] = norms

scores_dict_3 = {}
for idx, row in cluster_indices['indices'].items():
    indices_list = row.replace("'", "").split(", ")
    D_1 = np.zeros(n1)
    for indices in indices_list:
        matrix_index, col_index = indices.split('_')
        matrix_index = int(matrix_index) - 1
        col_index = int(col_index) - 1
        key = keys[matrix_index]
        D_1 += scores_dict_2[key][:, col_index]
    D_1 /= cluster_indices['count'][idx]
    scores_dict_3[idx] = D_1

scores_1 = np.column_stack([scores_dict_3[key] for key in scores_dict_3])

# Phase I T2 control chart
m, p = scores_1.shape
M = np.mean(scores_1, axis=0)  # mean
S = np.cov(scores_1, rowvar=False)  # covariance matrix
Pinv = np.linalg.inv(S)
T2_values_1 = []
for i in range(m):
    x = scores_1[i]
    diff = x - M
    T2 = np.dot(np.dot(diff.T, np.linalg.inv(S)), diff)
    T2_values_1.append(T2)
T2_values_1 = np.array(T2_values_1)

# adjust control limit so that ARL0 = 100
CL = find_adjusted_CL(CL_lower=20, CL_upper=30)

c_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for c in c_list:
    RL_list = []
    for seed in seeds_list:

        # phase 2
        n2 = 1000

        # get new data in control
        data_dict_2 = oc_data_generation(c, n2, seed * 3)

        scores_dict_4 = {}
        for key in keys:
            scores_dict_4[key] = data_dict_2[key] @ eigenvectors_dict_1[key]

        for key, matrix in scores_dict_4.items():
            scores_dict_4[key] = matrix / l_dict[key]

        scores_dict_5 = {}
        for idx, row in cluster_indices['indices'].items():
            indices_list = row.replace("'", "").split(", ")
            D_2 = np.zeros(n2)
            for indices in indices_list:
                matrix_index, col_index = indices.split('_')
                matrix_index = int(matrix_index) - 1
                col_index = int(col_index) - 1
                key = keys[matrix_index]
                D_2 += scores_dict_4[key][:, col_index]
            D_2 /= cluster_indices['count'][idx]
            scores_dict_5[idx] = D_2

        scores_2 = np.column_stack([scores_dict_5[key] for key in scores_dict_5])

        # Phase II T2 control chart
        diffs_2 = scores_2 - M
        T2_values_2 = np.sum((diffs_2 @ Pinv) * diffs_2, axis=1)

        outlier_found = False
        for i, value in enumerate(T2_values_2):
            if value > CL:
                RL = i + 1
                RL_list.append(RL)
                # print(f"First outlier found at index: {RL}")
                outlier_found = True
                break
        # if not outlier_found:
        #     print("No outliers found")

    ARL = np.mean(RL_list)
    print(f'{ARL}')
print('--------')
