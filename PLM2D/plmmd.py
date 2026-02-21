import copy
import csv
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def hc_distance(a1, a2):
    """
    Distance metric that measures the similarity between any two score vectors.
    :param a1: A unit vector.
    :param a2: A unit vector.
    """
    dist1 = 1 - np.dot(a1, a2)  # cosine distance
    dist2 = 1 - np.dot(a1, -a2)
    return min(dist1, dist2)


def per_min_distance(Q_datas):
    """
    Permute the score matrices once and calculate the minimum distance among all the score vectors from the new
    score matrices.
    :param Q_datas: A dictionary consisting of orthogonal score matrices.
    """
    # Shuffle the row vectors of each score matrices$
    Q_datas_shuffle = copy.deepcopy(Q_datas)
    for key in Q_datas_shuffle:
        np.random.shuffle(Q_datas_shuffle[key])
    stacked_matrix = np.hstack(list(Q_datas_shuffle.values()))

    # Calculate the minimum distance among all the score vectors from the new score matrices
    min_distance = np.inf
    num_cols = stacked_matrix.shape[1]
    for k in range(num_cols):
        dists = np.minimum(1 - np.dot(stacked_matrix[:, k], stacked_matrix),
                           1 - np.dot(stacked_matrix[:, k], -stacked_matrix))
        dists[k] = np.inf
        min_distance = min(min_distance, np.min(dists))

    return min_distance


def finding_threshold(datas):
    """
    Finding the cut-off value after n_perm times permutation.
    :param datas: A dictionary consisting of $\Gamma$.
    """
    # QR decomposition
    Q_datas = {}
    for key, value in datas.items():
        Q_data, _ = np.linalg.qr(value)
        Q_datas[key] = Q_data

    # calculate percentile
    min_distances = []
    for i in range(10000):  # 10000 permutations
        min_distance = per_min_distance(Q_datas)
        min_distances.append(min_distance)
    percentile = np.percentile(min_distances, 10)  # 10th percentile

    return percentile


def hc_all(datas, distance_threshold, filename, save_path, pc_numbers):
    """
    Perform the hierarchical clustering to $\Gamma$.
    :param pc_numbers: A list of principal component numbers.
    :param save_path: The path to save the dendrogram.
    :param datas: A dictionary consisting of $\Gamma$.
    :param distance_threshold: The cut-off value.
    :param filename: A CSV file storing clustering results.
    :return: A dictionary storing clustering results.
    """

    # QR decomposition
    Q_datas = {}
    for key, value in datas.items():
        Q_data, _ = np.linalg.qr(value)
        Q_datas[key] = Q_data

    # horizontally stack all the orthogonal matrices
    stacked_matrix = np.hstack(list(Q_datas.values()))

    labels = []
    for pc_number, count in enumerate(pc_numbers, start=1):
        for element_num in range(1, count + 1):
            label = f'{pc_number}_{element_num}'
            labels.append(label)

    linkage_result = linkage(stacked_matrix.T, metric=hc_distance, method='complete')
    linkage_result[:, 2] = np.where(linkage_result[:, 2] < 1e-10, 0, linkage_result[:, 2])

    # # dendrogram
    # plt.figure(figsize=(20, 4))
    # dendrogram(linkage_result, labels=labels, color_threshold=0)
    # plt.axhline(y=distance_threshold, color='grey', linestyle='--', linewidth=2, label='cut-off line')
    # plt.xlabel("Score Vector", fontsize=12)
    # plt.ylabel("Distance", fontsize=12)
    # plt.tick_params(axis='both', labelsize=12)
    # plt.legend(loc='best', fontsize=12)
    # plt.tight_layout()
    # plt.savefig(save_path + f'{filename}.png', dpi=300)
    # plt.close()

    # get clusters according to distance threshold
    cluster = fcluster(linkage_result, t=distance_threshold, criterion='distance')
    cluster_indices = {}
    for j, cluster_label in enumerate(cluster):
        if cluster_label not in cluster_indices:
            cluster_indices[cluster_label] = []
        cluster_indices[cluster_label].append(labels[j])

    # # save to csv
    # with open(save_path + f'{filename}.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['indices', 'count'])
    #     for indices in cluster_indices.values():
    #         count = len(indices)
    #         writer.writerow([indices, count])

    return cluster_indices


def dict_difference(dict1, dict2):
    """
    Computes the sum of squared Frobenius norms of the differences between corresponding matrices
    in two dictionaries.
    """
    diff = 0
    for (key1, value1), (key2, value2) in zip(dict1.items(), dict2.items()):
        diff += np.sum((value1 - value2) ** 2)
    return diff


def execute_ADMM(B_dict, A_dict, train_datas, keys, pc_numbers, filename='structure_1', rho=1e8, n=50):
    """
    The ADMM algorithm that alternatively iterates the score matrices (A), the Lagrange multiplier matrices (U) and
    the reformulation matrices ($\Gamma$) when B is given.
    :param n: sample size
    :param rho: tuning parameter obtained by grid search
    :param train_datas: A dictionary consisting of training data.
    :param A_dict: A dictionary consisting of score matrices (A).
    :param B_dict: A dictionary consisting of loading matrices (B)
    :param filename: The filename of a csv file storing the structure of score matrices after hierarchical clustering.
    :return: A, D (a matrix of $\gamma_q, q=1, \dots, g$), F ($\Gamma$), diff1, diff2.
    """
    init_U_dict = {}  # initial Lagrangian multiplier U
    for key, pc_number in zip(keys, pc_numbers):
        U = np.zeros((n, pc_number))
        init_U_dict[key] = U

    # Iteration settings
    max_iterations = 1000  # Maximum iterations
    dual_residual = 1e-5  # threshold of dual residual
    primal_residual = 1e-5  # threshold of primal residual

    # Initialize D according to the initial structure
    cluster_indices = pd.read_csv(f"{filename}.csv")  # get structure
    cluster_indices['indices'] = cluster_indices['indices'].str.strip('[]')
    init_D_dict = {}  # initial dict of $\gamma$
    for idx, row in cluster_indices['indices'].items():
        indices_list = row.replace("'", "").split(", ")
        D = np.zeros(n)
        for indices in indices_list:
            matrix_index, col_index = indices.split('_')
            matrix_index = int(matrix_index) - 1
            col_index = int(col_index) - 1
            key = keys[matrix_index]
            D += A_dict[key][:, col_index] + init_U_dict[key][:, col_index]
        D /= cluster_indices['count'][idx]
        init_D_dict[idx] = D

    # Initialize the dict of $\Gamma^{(j)}, j=1, \dots, p$
    init_F_dict = {}
    init_E_dict = {}
    for idx, row in cluster_indices['indices'].items():
        indices_list = row.replace("'", "").split(", ")
        for indices in indices_list:
            matrix_index, col_index = indices.split('_')
            if matrix_index not in init_E_dict:
                init_E_dict[matrix_index] = {}
            init_E_dict[matrix_index][col_index] = init_D_dict[idx]
    sorted_E_dict = {
        outer_key: dict(sorted(inner_dict.items(), key=lambda item: int(item[0])))
        for outer_key, inner_dict in init_E_dict.items()
    }  # sort the dictionary by inner key
    sorted_E_dict = dict(sorted(sorted_E_dict.items(), key=lambda x: int(x[0])))  # sort the dictionary by outer key
    init_E_dict = sorted_E_dict
    for outer_key, inner_dict in init_E_dict.items():  # concatenate column vectors to matrix
        df = pd.DataFrame(inner_dict)
        init_F_dict[outer_key] = df.values
    c_F_dict = {}  # change keys
    for (outer_key, value), key in zip(init_F_dict.items(), keys):
        c_F_dict[key] = value
    init_F_dict = c_F_dict

    # Deep copy the initial dictionaries to ensure they remain unchanged outside the function
    A_dict = copy.deepcopy(A_dict)  # from outer space
    U_dict = copy.deepcopy(init_U_dict)  # from outer space
    D_dict = copy.deepcopy(init_D_dict)  # from outer space
    F_dict = copy.deepcopy(init_F_dict)

    for iteration in range(max_iterations):
        # step1: update A
        new_A_dict = {}
        C_dict = {}
        for key in keys:
            C = 2 * train_datas[key] @ B_dict[key] - U_dict[key] + rho * F_dict[key]
            C_dict[key] = C
            M, S, Nt = np.linalg.svd(C_dict[key], full_matrices=0)
            A = M @ Nt
            new_A_dict[key] = A
        A_dict.update(new_A_dict)

        # step 2: update $\gamma$
        new_D_dict = {}
        for idx, row in cluster_indices['indices'].items():
            indices_list = row.replace("'", "").split(", ")
            D = np.zeros(n)
            for indices in indices_list:
                matrix_index, col_index = indices.split('_')
                matrix_index = int(matrix_index) - 1
                col_index = int(col_index) - 1
                key = keys[matrix_index]
                D += A_dict[key][:, col_index] + 1 / rho * U_dict[key][:, col_index]
            D /= cluster_indices['count'][idx]
            new_D_dict[idx] = D
        diff1 = dict_difference(new_D_dict, D_dict)  # dual residual
        D_dict.update(new_D_dict)

        # use $\gamma$ to represent $\Gamma$
        E_dict = {}
        new_F_dict = {}
        for idx, row in cluster_indices['indices'].items():
            indices_list = row.replace("'", "").split(", ")
            for indices in indices_list:
                matrix_index, col_index = indices.split('_')
                if matrix_index not in E_dict:
                    E_dict[matrix_index] = {}
                E_dict[matrix_index][col_index] = D_dict[idx]

        sorted_E_dict = {
            outer_key: dict(sorted(inner_dict.items(), key=lambda item: int(item[0])))
            for outer_key, inner_dict in E_dict.items()
        }  # sort the dictionary by inner key
        sorted_E_dict = dict(sorted(sorted_E_dict.items(), key=lambda x: int(x[0])))  # sort the dictionary by outer key
        E_dict = sorted_E_dict

        for outer_key, inner_dict in E_dict.items():  # concatenate column vectors to matrix
            df = pd.DataFrame(inner_dict)
            new_F_dict[outer_key] = df.values
        c_F_dict = {}  # change keys
        for (outer_key, value), key in zip(new_F_dict.items(), keys):
            c_F_dict[key] = value
        new_F_dict = c_F_dict
        F_dict.update(new_F_dict)
        diff2 = dict_difference(F_dict, A_dict)  # primal residual

        # Stopping criterion
        if diff1 < dual_residual and diff2 < primal_residual:
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

    return A_dict, D_dict, F_dict, diff1, diff2


def update_B(A_dict, train_datas, keys, best_params, beta=1e-5):
    """
    Update the loading matrices (B) when A is given.
    :param train_datas: A dictionary consisting of training data.
    :param A_dict: A dictionary consisting of the score matrices (A).
    :param beta: The coefficient $\eta$.
    :return: B_dict.
    """

    # The second-order derivative of a function
    matrix_size = 70  # time points
    R = np.eye(matrix_size)
    R += np.eye(matrix_size, k=1) * -2
    R += np.eye(matrix_size, k=2) * 1
    R = R[0:68, :]

    G_dict = {}
    for best_param, key in zip(best_params, keys):
        G = np.eye(70) + beta * best_param * R.T @ R
        G_dict[key] = G

    B_dict = {}
    for key in keys:
        B = np.linalg.inv(G_dict[key]) @ train_datas[key].T @ A_dict[key]
        B_dict[key] = B

    return B_dict


def parameter_learning(scores_dict, eigenvectors_dict, train_datas, keys, pc_numbers, best_params, filename='structure_1', beta=1e-5):
    """
    The final ADMM algorithm that iterates the score matrices (A), the Lagrange multiplier matrices (U),
    the reformulation matrices ($\Gamma$) and the loading matrices (B).
    :param train_datas: A dictionary consisting of training data.
    :param eigenvectors_dict: A dictionary consisting of the eigenvectors of score matrices (G).
    :param scores_dict: A dictionary consisting of the score matrices (A).
    :param filename: The filename of a csv file storing the structure of score matrices after hierarchical clustering.
    :param beta: The coefficient $\eta$.
    :return: F_dict, A_dict, B_dict, D_dict.
    """
    tolerance = 1e-4  # tolerance of parameter learning $\epsilon^t$
    max_iterations = 1000
    iteration = 0
    converged = False
    previous_A_dict = copy.deepcopy(scores_dict)  # from outer space
    B_dict = copy.deepcopy(eigenvectors_dict)  # from outer space

    for iteration in range(max_iterations):
        A_dict, D_dict, F_dict, diff1, diff2 = execute_ADMM(B_dict, previous_A_dict, train_datas, keys, pc_numbers, filename)
        error = dict_difference(A_dict, previous_A_dict)
        # print(diff1)
        # print(diff2)
        # print(error)

        if error < tolerance:
            converged = True
            break
        B_dict = update_B(A_dict, train_datas, keys, best_params, beta)
        previous_A_dict.update(A_dict)

    # if converged:
    #     print(f'The parameter learning process converged in {iteration + 1} iterations')
    # else:
    #     print(f'The parameter learning process failed to converge within {max_iterations} iterations')

    return F_dict, A_dict, B_dict, D_dict


def compare_structure(dict1, dict2):
    """
    Compare the current structure with that from the previous iteration.
    """
    values1 = set(tuple(sorted(lst)) for lst in dict1.values())
    values2 = set(tuple(sorted(lst)) for lst in dict2.values())

    return values1 == values2





def draw_scree_plot(train_datas, save_path):
    """
    perform initial pca to determine pc_numbers, scores_dict, eigenvectors_dict
    """
    # PCA
    eigenvalues = {}
    ratios = {}
    for key, value in train_datas.items():
        pca = PCA()
        pca.fit(value)
        eigenvalue = pca.explained_variance_
        ratio = eigenvalue / np.sum(eigenvalue)
        eigenvalues[key] = eigenvalue
        ratios[key] = ratio

    # scree plot
    for key, values in eigenvalues.items():
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(values) + 1), values, marker='o',
                 linestyle='-', color='b')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        plt.title(f'Scree Plot for {key}')
        plt.savefig(f'{save_path}/scree_plot_{key}.png')
        plt.close()
        plt.show()


def perform_initial_pca(train_datas, pc_numbers):
    """
    perform initial pca to determine pc_numbers, scores_dict, eigenvectors_dict
    """
    scores_dict = {}
    eigenvectors_dict = {}

    for idx, (key, value) in enumerate(train_datas.items()):
        k = pc_numbers[idx]
        pca = PCA(n_components=k)
        score = pca.fit_transform(value)
        eigenvector = pca.components_.T
        scores_dict[key] = score
        eigenvectors_dict[key] = eigenvector

    return scores_dict, eigenvectors_dict


if __name__ == "__main__":
    path = ""
    ori_data = np.load(path + "MTR.npy")  # shape:(N days, P stations, T time points)
    df = pd.read_excel(path + "station_indices.xlsx", header=None)
    indices = df.index

    datas = {}
    keys = []
    for i in indices:
        data = ori_data[:, i, :]
        data_name = f'data_{df[1][i]}'
        datas[data_name] = data
        keys.append(data_name)

    days = range(194)
    train_days = days[0:116]  # 60%-116days
    validation_days = days[116:155]  # 20%-39days
    test_days = days[155:194]  # 20%-39days

    train_datas = {}
    for key, value in datas.items():
        train_data = value[train_days, :]
        train_datas[key] = train_data

    # column-centered
    for key, value in train_datas.items():
        train_data_mean = np.mean(value, axis=0)
        train_datas[key] = np.subtract(value, train_data_mean)

    # 17 stations
    rho = 1e8  # penalty parameter of ADMM
    best_params = [53570, 27031, 3215, 364986, 1945, 555817, 437424, 35036, 324114, 511958, 135050, 550696, 10040,
                   612431, 114870, 67646, 2986]  # tuning parameters for each station

    draw_scree_plot(train_datas, "./plots")
    pc_numbers = [8, 8, 8, 7, 5, 7, 8, 9, 8, 6, 8, 6, 8, 8, 7, 9, 10]  # manual selection example
    scores_dict, eigenvectors_dict = perform_initial_pca(train_datas, pc_numbers, "./")

    max_k = 15
    k = 0
    beta = 1e-5
    converged = False
    df = pd.read_csv(path + "structure_1.csv")  # initial structure
    previous_s = {i: eval(row) for i, row in enumerate(df['indices'])}
    previous_filename = 'structure_1'
    for k in range(max_k):
        F, A, B, D = parameter_learning(
            scores_dict=scores_dict,
            eigenvectors_dict=eigenvectors_dict,
            train_datas=train_datas,
            keys=keys,
            pc_numbers=pc_numbers,
            best_params=best_params,
            filename=previous_filename,
            beta=beta
        )
        percentile = finding_threshold(F)
        s = hc_all(F, percentile, f'structure_{k + 2}', save_path="./", pc_numbers=pc_numbers)

        if compare_structure(s, previous_s) == 1:
            print('The structures are identical.')
            # orthogonalization
            V = {}
            for key, value in B.items():
                Q, _ = np.linalg.qr(value)
                V[key] = Q

            converged = True
            break

        else:
            print('The structures are different.')
            previous_s = s
            previous_filename = f'structure_{k + 2}'

    if converged:
        print(f'The algorithm converged in {k + 1} iterations')
    else:
        print(f'The algorithm failed to converge within {max_k} iterations')

    # save
    df = pd.DataFrame(D)
    df = df.T
    df.to_csv(path + "D.csv")
    for key, matrix in F.items():
        df = pd.DataFrame(matrix)
        file_name = path + f"F_{key}_final.csv"
        df.to_csv(file_name, index=False)
    for key, matrix in A.items():
        df = pd.DataFrame(matrix)
        file_name = path + f"A_{key}_final.csv"
        df.to_csv(file_name, index=False)
    for key, matrix in B.items():
        df = pd.DataFrame(matrix)
        file_name = path + f"B_{key}_final.csv"
        df.to_csv(file_name, index=False)
