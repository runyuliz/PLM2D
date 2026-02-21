import os
import numpy as np
import pandas as pd
import csv
from data_generation import data_generation
import plmmd

save_path = ''
keys = ['data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW']
pc_numbers = [7, 7, 6, 6, 8]
best_params = [364986, 555817, 511958, 550696, 612431]
n_list = [50, 100, 200, 300, 400, 500]
seeds_list = [1501, 2586, 2653, 1055, 705, 106, 589, 2468, 2413, 1600,
              2464, 228, 915, 794, 3021, 3543, 1073, 3351, 1744, 1084,
              926, 3049, 1117, 642, 4767, 501, 4066, 333, 4684, 486,
              1962, 393, 4842, 4866, 1755, 2515, 3585, 4315, 4966, 2099,
              3599, 4121, 29, 65, 838, 3906, 3773, 4635, 3161, 2659,
              4615, 4628, 2451, 2846, 1144, 3078, 1103, 168, 1670, 2570,
              2377, 4395, 4257, 3862, 23, 2633, 3340, 2215, 3682, 4724,
              1907, 84, 227, 296, 1001, 2138, 711, 2801, 2527, 3752,
              3321, 3181, 4183, 1615, 3024, 1413, 763, 655, 4797, 3849,
              4153, 468, 157, 1295, 497, 4740, 2940, 3456, 373,
              79]

# To verify the workflow first, uncomment the code block below
# n = 50
# seed = 1501
# n_list = [n]
# seeds_list = [seed]

for n in n_list:
    for seed in seeds_list:
        data_dict = data_generation(n, seed)

        scores_dict,  eigenvectors_dict = plmmd.perform_initial_pca(data_dict, pc_numbers)
        percentile = plmmd.finding_threshold(scores_dict)
        cluster_indices = plmmd.hc_all(scores_dict, percentile, 'structure_1', save_path, pc_numbers)

        # algorithm
        max_k = 15
        k = 0
        beta = 1e-5
        MSE_list = []
        RMSE_list = []
        MAE_list = []
        converged = False
        df = pd.read_csv("structure_1.csv")
        previous_s = {i: eval(row) for i, row in enumerate(df['indices'])}
        print(previous_s)
        previous_filename = 'structure_1'
        for k in range(max_k):
            F, A, B, D = plmmd.parameter_learning(scores_dict, eigenvectors_dict, data_dict, keys, pc_numbers, best_params, previous_filename)
            percentile = plmmd.finding_threshold(F)
            s = plmmd.hc_all(F, percentile, f'structure_{k + 2}', save_path, pc_numbers)

            if plmmd.compare_structure(s, previous_s) == 1:
                converged = True
                print('The structures are identical.')

                V = {}
                for key, value in B.items():
                    Q, _ = np.linalg.qr(value)
                    V[key] = Q

                folder_path = save_path+f"n={n}, 100seeds/n={n}, seed={seed}/"
                os.makedirs(folder_path, exist_ok=True)

                with open(folder_path + f'n={n}, seed={seed}.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['indices', 'count'])
                    for indices in s.values():
                        count = len(indices)
                        writer.writerow([indices, count])

                for key, matrix in B.items():
                    df = pd.DataFrame(matrix)
                    file_name = folder_path + f"{key}_final.csv"
                    df.to_csv(file_name, index=False)
                break

            else:
                print('The structures are different.')
                previous_s = s
                previous_filename = f'structure_{k+2}'

        if converged:
            print(f'The algorithm converged in {k + 1} iterations')
        else:
            print(f'The algorithm failed to converge within {max_k} iterations')



