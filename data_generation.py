import numpy as np
import pandas as pd


def data_generation(n, seed):
    """
    generate data for simulations based on the eigenvectors from real case study
    :param n: sample size
    :param seed: random seed
    :return: a dictionary of simulated data for five chosen stations
    """
    keys = ['data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW']
    pc_numbers = [7, 7, 6, 6, 8]
    tru_eigenvectors_dict = {}
    for key in keys:
        B = pd.read_csv(
            f"eigenvectors/{key}.csv", header=None)
        tru_eigenvectors_dict[key] = B.values

    np.random.seed(seed)  # random seed

    scores_dict = {}
    for idx, key in enumerate(keys):
        A = np.zeros((n, pc_numbers[idx]))
        scores_dict[key] = A

    cluster_indices = pd.read_csv("structure_5stations.csv")  # get structure
    cluster_indices['indices'] = cluster_indices['indices'].str.strip('[]')

    for idx, row in cluster_indices['indices'].items():
        indices_list = row.replace("'", "").split(", ")
        score = np.random.normal(0, 1, n)
        for indices in indices_list:
            matrix_index, col_index = indices.split('_')
            matrix_index = int(matrix_index) - 1
            col_index = int(col_index) - 1
            key = keys[matrix_index]
            scores_dict[key][:, col_index] = score

    for matrix in scores_dict.values():
        cols = matrix.shape[1]
        for i in range(cols):
            magnitude = cols + 1 - i
            matrix[:, i] *= magnitude

    em_dict = {}
    for key in keys:
        error_matrix = np.random.normal(0, 1, (n, 70))
        em_dict[key] = error_matrix

    data_dict = {}
    for key in keys:
        data_dict[key] = scores_dict[key] @ tru_eigenvectors_dict[key].T + em_dict[key]

    return data_dict


def oc_data_generation(mean, n, seed):
    """ oc scenario 1"""
    keys = ['data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW']
    pc_numbers = [7, 7, 6, 6, 8]
    tru_eigenvectors_dict = {}
    for key in keys:
        B = pd.read_csv(
            f"eigenvectors/{key}.csv", header=None)
        tru_eigenvectors_dict[key] = B.values

    np.random.seed(seed)  # random seed
    cluster_indices = pd.read_csv("structure_5stations.csv")  # get structure
    cluster_indices['indices'] = cluster_indices['indices'].str.strip('[]')

    scores_dict = {}
    for idx, key in enumerate(keys):
        A = np.zeros((n, pc_numbers[idx]))
        scores_dict[key] = A

    for idx, row in cluster_indices['indices'].items():
        indices_list = row.replace("'", "").split(", ")
        score = np.random.normal(mean, 1, n)
        for indices in indices_list:
            matrix_index, col_index = indices.split('_')
            matrix_index = int(matrix_index) - 1
            col_index = int(col_index) - 1
            key = keys[matrix_index]
            scores_dict[key][:, col_index] = score

    for matrix in scores_dict.values():
        cols = matrix.shape[1]
        for i in range(cols):
            magnitude = cols + 1 - i
            matrix[:, i] *= magnitude

    em_dict = {}
    for key in keys:
        error_matrix = np.random.normal(0, 1, (n, 70))
        em_dict[key] = error_matrix

    data_dict = {}
    for key in keys:
        data_dict[key] = scores_dict[key] @ tru_eigenvectors_dict[key].T + em_dict[key]

    return data_dict


def oc2_data_generation(mean, n, seed, k):
    """ oc scenario 2-4"""
    keys = ['data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW']
    pc_numbers = [7, 7, 6, 6, 8]
    tru_eigenvectors_dict = {}
    for key in keys:
        B = pd.read_csv(
            f"eigenvectors/{key}.csv", header=None)
        tru_eigenvectors_dict[key] = B.values

    np.random.seed(seed)  # random seed
    cluster_indices = pd.read_csv("structure_5stations.csv")  # get structure
    cluster_indices['indices'] = cluster_indices['indices'].str.strip('[]')

    scores_dict = {}
    for idx, key in enumerate(keys):
        A = np.zeros((n, pc_numbers[idx]))
        scores_dict[key] = A

    for idx, row in cluster_indices['indices'].items():
        indices_list = row.replace("'", "").split(", ")
        score_oc = np.random.normal(mean, 1, n)
        score_ic = np.random.normal(0, 1, n)
        for indices in indices_list:
            matrix_index, col_index = indices.split('_')
            matrix_index = int(matrix_index) - 1
            col_index = int(col_index) - 1
            key = keys[matrix_index]
            if idx == k - 1:
                scores_dict[key][:, col_index] = score_oc
            else:
                scores_dict[key][:, col_index] = score_ic

    for matrix in scores_dict.values():
        cols = matrix.shape[1]
        for i in range(cols):
            magnitude = cols + 1 - i
            matrix[:, i] *= magnitude

    em_dict = {}
    for key in keys:
        error_matrix = np.random.normal(0, 1, (n, 70))
        em_dict[key] = error_matrix

    data_dict = {}
    for key in keys:
        data_dict[key] = scores_dict[key] @ tru_eigenvectors_dict[key].T + em_dict[key]

    return data_dict


def oc3_data_generation(mean, n, seed, k1, k2):
    """ oc scenario 5-6"""
    keys = ['data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW']
    pc_numbers = [7, 7, 6, 6, 8]
    tru_eigenvectors_dict = {}
    for key in keys:
        B = pd.read_csv(
            f"eigenvectors/{key}.csv", header=None)
        tru_eigenvectors_dict[key] = B.values

    np.random.seed(seed)  # random seed
    cluster_indices = pd.read_csv("structure_5stations.csv")  # get structure
    cluster_indices['indices'] = cluster_indices['indices'].str.strip('[]')

    scores_dict = {}
    for idx, key in enumerate(keys):
        A = np.zeros((n, pc_numbers[idx]))
        scores_dict[key] = A

    for idx, row in cluster_indices['indices'].items():
        indices_list = row.replace("'", "").split(", ")
        score_oc = np.random.normal(mean, 1, n)
        score_ic = np.random.normal(0, 1, n)
        for indices in indices_list:
            matrix_index, col_index = indices.split('_')
            matrix_index = int(matrix_index) - 1
            col_index = int(col_index) - 1
            key = keys[matrix_index]
            if idx == k1 - 1 or idx == k2 - 1:
                scores_dict[key][:, col_index] = score_oc
            else:
                scores_dict[key][:, col_index] = score_ic

    for matrix in scores_dict.values():
        cols = matrix.shape[1]
        for i in range(cols):
            magnitude = cols + 1 - i
            matrix[:, i] *= magnitude

    em_dict = {}
    for key in keys:
        error_matrix = np.random.normal(0, 1, (n, 70))
        em_dict[key] = error_matrix

    data_dict = {}
    for key in keys:
        data_dict[key] = scores_dict[key] @ tru_eigenvectors_dict[key].T + em_dict[key]

    return data_dict


if __name__ == "__main__":
    print("--- Generating sample data ---")
    n_sample = 50
    test_seed = 4724
    try:
        data = data_generation(n_sample, test_seed)
        for k, v in data.items():
            print(f"Station: {k}\n{v}\n")
        print("\nData generation successful!")
    except FileNotFoundError as e:
        print(f"Error: Please ensure the required CSV files exist. {e}")

