import numpy as np
import pandas as pd
import copy
import os
from sklearn.decomposition import PCA
from tensorly.decomposition import tucker, parafac

from data_generation import data_generation


def run_idlfm(seed, path, n=50):
    def dict_difference(dict1, dict2):
        # Frobenius norm
        diff = 0
        for (key1, value1), (key2, value2) in zip(dict1.items(), dict2.items()):
            diff += np.sum((value1 - value2) ** 2)
        return diff

    def execute_ADMM(B_dict):
        # iteration settings
        max_iterations = 1000
        dual_residual = 1e-5
        primal_residual = 1e-5
        converged = False

        all_matrices = np.stack(list(scores_dict.values()))
        init_D = np.mean(all_matrices, axis=0)

        init_F_dict = {}  # initial dict of gamma(parameter)
        for key in keys:
            init_F_dict[key] = init_D.copy()

        # Deep copy the initial dictionaries to ensure they remain unchanged outside the function
        A_dict = copy.deepcopy(scores_dict)  # from outer space
        U_dict = copy.deepcopy(init_U_dict)  # from outer space
        D = copy.deepcopy(init_D)
        F_dict = copy.deepcopy(init_F_dict)

        for iteration in range(max_iterations):
            # step1: update A
            new_A_dict = {}
            C_dict = {}
            for key in keys:
                C = 2 * data_dict[key] @ B_dict[key] - U_dict[key] + rho * F_dict[key]
                C_dict[key] = C
                M, S, Nt = np.linalg.svd(C_dict[key], full_matrices=0)
                A = M @ Nt
                new_A_dict[key] = A
            A_dict.update(new_A_dict)

            # step 2: update Gamma
            all_matrices_new = np.stack(list(A_dict.values()))
            new_D = np.mean(all_matrices_new, axis=0)

            diff1 = np.mean((new_D - D) ** 2)
            D = new_D

            # use gamma to represent Gamma
            new_F_dict = {}
            for key in keys:
                new_F_dict[key] = D.copy()
            F_dict.update(new_F_dict)

            diff2 = dict_difference(F_dict, A_dict)
            if diff1 < dual_residual and diff2 < primal_residual:
                converged = True
                break

            # step 3: update U
            new_U_dict = {}
            for key in U_dict.keys():
                new_U_dict[key] = U_dict[key] + rho * (A_dict[key] - F_dict[key])
            U_dict.update(new_U_dict)

        # if converged:
        #     print(f'ADMM converged in {iteration + 1} iterations')
        # else:
        #     print(f'ADMM failed to converge within {max_iterations} iterations')

        return A_dict, D, F_dict, diff1, diff2

    def update_B(A_dict):
        B_dict = {}
        for key in keys:
            B = data_dict[key].T @ A_dict[key]
            B_dict[key] = B

        return B_dict

    def parameter_learning():
        tolerance = 1e-4
        max_iterations = 1000
        iteration = 0
        converged = False
        previous_A_dict = copy.deepcopy(scores_dict)  # from outer space
        B_dict = copy.deepcopy(eigenvectors_dict)  # from outer space

        for iteration in range(max_iterations):
            A_dict, D, F_dict, diff1, diff2 = execute_ADMM(B_dict)
            error = dict_difference(A_dict, previous_A_dict)

            # print(diff1)
            # print(diff2)
            # print(error)

            if error < tolerance:
                converged = True
                break
            B_dict = update_B(A_dict)
            previous_A_dict.update(A_dict)

        # if converged:
        #     print(f'The parameter learning process converged in {iteration + 1} iterations')
        # else:
        #     print(f'The parameter learning process failed to converge within {max_iterations} iterations')

        return F_dict, A_dict, B_dict, D

    print(f"Processing seed: {seed}")
    rho = 1e8
    keys = ['data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW']
    k = 10
    pc_numbers = [k] * len(keys)

    data_dict = data_generation(n, seed)

    eigenvectors_dict = {}
    scores_dict = {}
    for key, pc_number in zip(keys, pc_numbers):
        pca = PCA(n_components=pc_number)
        scores_dict[key] = pca.fit_transform(data_dict[key])
        eigenvectors_dict[key] = pca.components_.T

    init_U_dict = {}
    for key, pc_number in zip(keys, pc_numbers):
        U = np.zeros((n, pc_number))
        init_U_dict[key] = U

    F, A, B, D = parameter_learning()

    out_dir = path + f"IDLFM/n={n}, seed={seed}/"
    os.makedirs(out_dir, exist_ok=True)
    for key, matrix in B.items():
        df = pd.DataFrame(matrix)
        file_name = out_dir + f"{key}.csv"
        df.to_csv(file_name, index=False)


def run_seq(seed, path, n=50):
    def smoothPCA(X, pc_number, alpha):
        U, S, Vt = np.linalg.svd(X)
        V = Vt.T
        A = V[:, 0:pc_number]
        C = X.T @ X

        matrix_size = 70
        R = np.eye(matrix_size)
        R += np.eye(matrix_size, k=1) * -2
        R += np.eye(matrix_size, k=2) * 1
        R = R[0:68, :]

        max_iterations = 1000
        tolerance = 1e-4
        converged = False

        for iteration in range(max_iterations):
            # Step 2
            a_dict = {}
            for i in range(pc_number):
                a_dict[f'a{i + 1}'] = A[:, i]

            y_dict = {}
            for i in range(pc_number):
                y_dict[f'y{i + 1}'] = X @ a_dict[f'a{i + 1}']

            b_dict = {}
            for i in range(pc_number):
                b_dict[f'b{i + 1}'] = np.linalg.inv(C + alpha * R.T @ R) @ X.T @ y_dict[f'y{i + 1}']  # same alpha

            # Step 3
            b_values = [b_dict[f'b{i + 1}'] for i in range(pc_number)]
            B = np.column_stack(b_values)

            U, S, Vt = np.linalg.svd(C @ B, full_matrices=0)

            # Reconstruction of matrix A
            new_A = U @ Vt

            # Check for convergence
            if np.linalg.norm(new_A - A) < tolerance:
                converged = True
                break

            A = new_A

        if not converged:
            print("Did not converge within the maximum number of iterations.")
        '''
        for i in range(pc_number):
            b_dict[f'b{i + 1}'] = b_dict[f'b{i + 1}'] / np.linalg.norm(b_dict[f'b{i + 1}'])
        v_values = [b_dict[f'b{i + 1}'] for i in range(pc_number)]
        V = np.column_stack(v_values)
        '''
        return B

    keys = ['data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW']
    pc_numbers = [7, 7, 6, 6, 8]
    best_params = [364986, 555817, 511958, 550696, 612431]

    data_dict = data_generation(n, seed)

    # execute smoothPCA
    eigenvectors_dict = {}
    scores_dict = {}
    for key, value in data_dict.items():
        index = list(data_dict.keys()).index(key)
        B = smoothPCA(value, pc_numbers[index], best_params[index])
        V, _ = np.linalg.qr(B)
        eigenvectors_dict[key] = V
        scores_dict[key] = value @ V

    cluster_indices = pd.read_csv(f"pca/{n}_{seed}_structure_pca.csv")
    cluster_indices['indices'] = cluster_indices['indices'].str.strip('[]')

    D_dict = {}  # initial dict of gamma(parameter)
    for idx, row in cluster_indices['indices'].items():
        indices_list = row.replace("'", "").split(", ")
        D = np.zeros(n)
        for indices in indices_list:
            matrix_index, col_index = indices.split('_')
            matrix_index = int(matrix_index) - 1
            col_index = int(col_index) - 1
            key = keys[matrix_index]
            D += scores_dict[key][:, col_index]
        D /= cluster_indices['count'][idx]
        D_dict[idx] = D

    init_E_dict = {}
    A_dict = {}
    for idx, row in cluster_indices['indices'].items():
        indices_list = row.replace("'", "").split(", ")
        for indices in indices_list:
            matrix_index, col_index = indices.split('_')
            if matrix_index not in init_E_dict:
                init_E_dict[matrix_index] = {}
            init_E_dict[matrix_index][col_index] = D_dict[idx]

    sorted_E_dict = {
        outer_key: dict(sorted(inner_dict.items(), key=lambda item: int(item[0])))
        for outer_key, inner_dict in init_E_dict.items()
    }  # sort the dictionary by inner key
    sorted_E_dict = dict(sorted(sorted_E_dict.items(), key=lambda x: int(x[0])))  # sort the dictionary by outer key
    init_E_dict = sorted_E_dict
    for outer_key, inner_dict in init_E_dict.items():  # concatenate column vectors to matrix
        df = pd.DataFrame(inner_dict)
        A_dict[outer_key] = df.values
    c_F_dict = {}  # change keys
    for (outer_key, value), key in zip(A_dict.items(), keys):
        c_F_dict[key] = value
    A_dict = c_F_dict

    B_dict = {}
    for key in keys:
        M = data_dict[key].T @ A_dict[key]
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        B = U @ Vt
        B_dict[key] = B

    for key, matrix in B_dict.items():
        df = pd.DataFrame(matrix)
        file_name = path + f"seq/n={n}, seed={seed}/{key}.csv"
        df.to_csv(file_name, index=False)


def run_vpca(seed, path, n=50):
    data_dict = data_generation(n, seed)
    VPCA_data = np.hstack(list(data_dict.values()))

    # get data in control
    pca = PCA(n_components=10)
    pca.fit(VPCA_data)
    eigenvectors = pca.components_.T

    df = pd.DataFrame(eigenvectors)
    file_name = path + f"vpca/n={n}, seed={seed}.csv"
    df.to_csv(file_name, index=False)


def run_mfpca(seed, path, n=50):
    data_dict = data_generation(n, seed)
    MFPCA_data = np.vstack(list(data_dict.values()))
    pca = PCA(n_components=7)
    pca.fit(MFPCA_data)
    eigenvectors = pca.components_.T
    df = pd.DataFrame(eigenvectors)
    file_name = path + f"mfpca/n={n}, seed={seed}.csv"
    df.to_csv(file_name, index=False)


def run_fpca(seed, path, n=50):
    def smoothPCA(X, pc_number, alpha):
        U, S, Vt = np.linalg.svd(X)
        V = Vt.T
        A = V[:, 0:pc_number]
        C = X.T @ X

        matrix_size = 70
        R = np.eye(matrix_size)
        R += np.eye(matrix_size, k=1) * -2
        R += np.eye(matrix_size, k=2) * 1
        R = R[0:68, :]

        max_iterations = 1000
        tolerance = 1e-4
        converged = False

        for iteration in range(max_iterations):
            # Step 2
            a_dict = {}
            for i in range(pc_number):
                a_dict[f'a{i + 1}'] = A[:, i]

            y_dict = {}
            for i in range(pc_number):
                y_dict[f'y{i + 1}'] = X @ a_dict[f'a{i + 1}']

            b_dict = {}
            for i in range(pc_number):
                b_dict[f'b{i + 1}'] = np.linalg.inv(C + alpha * R.T @ R) @ X.T @ y_dict[f'y{i + 1}']  # same alpha

            # Step 3
            b_values = [b_dict[f'b{i + 1}'] for i in range(pc_number)]
            B = np.column_stack(b_values)

            U, S, Vt = np.linalg.svd(C @ B, full_matrices=0)

            # Reconstruction of matrix A
            new_A = U @ Vt

            # Check for convergence
            if np.linalg.norm(new_A - A) < tolerance:
                converged = True
                break

            A = new_A

        if not converged:
            print("Did not converge within the maximum number of iterations.")
        '''
        for i in range(pc_number):
            b_dict[f'b{i + 1}'] = b_dict[f'b{i + 1}'] / np.linalg.norm(b_dict[f'b{i + 1}'])
        v_values = [b_dict[f'b{i + 1}'] for i in range(pc_number)]
        V = np.column_stack(v_values)
        '''
        return B

    pc_numbers = [7, 7, 6, 6, 8]
    best_params = [364986, 555817, 511958, 550696, 612431]
    data_dict = data_generation(n, seed)
    eigenvectors_dict = {}
    for key, value in data_dict.items():
        index = list(data_dict.keys()).index(key)
        B = smoothPCA(value, pc_numbers[index], best_params[index])
        V, _ = np.linalg.qr(B)
        eigenvectors_dict[key] = V

    for key, matrix in eigenvectors_dict.items():
        df = pd.DataFrame(matrix)
        file_name = path + f"fpca/n={n}, seed={seed}/{key}.csv"
        df.to_csv(file_name, index=False)


def run_tucker(seed, path, n=50):
    keys = ['data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW']

    data_dict = data_generation(n, seed)
    matrices_list = [data_dict[key] for key in keys]
    transposed_matrices = [matrix.T for matrix in matrices_list]
    tensor = np.stack(transposed_matrices, axis=0)

    rank = [4, 5, n]
    core, factors = tucker(tensor, rank=rank)  # HOOI
    U1 = factors[0]  # (5, 4)
    U2 = factors[1]  # (70, 5)

    df1 = pd.DataFrame(U1)
    file_name_1 = path + f"tucker/n={n}, seed={seed}, U1.csv"
    df1.to_csv(file_name_1, index=False)

    df2 = pd.DataFrame(U2)
    file_name_2 = path + f"tucker/n={n}, seed={seed}, U2.csv"
    df2.to_csv(file_name_2, index=False)


def run_cp(seed, path, n=50):
    keys = ['data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW']
    rank = 20
    data_dict = data_generation(n, seed)
    matrices_list = [data_dict[key] for key in keys]
    transposed_matrices = [matrix.T for matrix in matrices_list]
    tensor = np.stack(transposed_matrices, axis=0)

    core, factors = parafac(tensor, rank=rank, normalize_factors=True, init='random')
    A, B, C = factors  # (5, 20) (70, 20) (n, 20)

    df1 = pd.DataFrame(A)
    file_name_1 = path + f"cp/n={n}, seed={seed}, A.csv"
    df1.to_csv(file_name_1, index=False)

    df2 = pd.DataFrame(B)
    file_name_2 = path + f"cp/n={n}, seed={seed}, B.csv"
    df2.to_csv(file_name_2, index=False)


seeds_list = [1501, 2586, 2653, 1055, 705, 106, 589, 2468, 2413, 1600,
              2464, 228, 915, 794, 3021, 3543, 1073, 3351, 1744, 1084,
              926, 3049, 1117, 642, 4767, 501, 4066, 333, 4684, 486,
              1962, 393, 4842, 4866, 1755, 2515, 3585, 4315, 4966, 2099,
              3599, 4121, 29, 65, 838, 3906, 3773, 4635, 3161, 2659,
              4615, 4628, 2451, 2846, 1144, 3078, 1103, 168, 1670, 2570,
              2377, 4395, 4257, 3862, 23, 2633, 3340, 2215, 3682, 4724,
              1907, 84, 227, 296, 1001, 2138, 711, 2801, 2527, 3752,
              3321, 3181, 4183, 1615, 3024, 1413, 763, 655, 4797, 3849,  # np.random.seed(42)
              4153, 468, 157, 1295, 497, 4740, 2940, 3456, 373,
              79]  # np.random.choice(range(5000), size=100, replace=False)
n_list = [50, 100, 200, 300, 400, 500]
path = ''
for n in n_list:
    for seed in seeds_list:
        run_fpca(seed, path, n)
        run_mfpca(seed, path, n)
        run_vpca(seed, path, n)
        run_tucker(seed, path, n)
        run_cp(seed, path, n)
        run_seq(seed, path, n)
        run_idlfm(seed, path, n)

