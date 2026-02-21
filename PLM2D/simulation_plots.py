"""
================================================================================
This script automates the visualization of reconstruction errors (RE) for various
methods in simulations.
It generates two types of plots (Figures 7-8 in the paper):
1. RE Boxplot over n
2. RE Boxplot when n=200

--------------------------------------------------------------------------------
REQUIRED INPUTS:
--------------------------------------------------------------------------------
1.  DATA FILES:
    Expected Method Folders:
    - 'idlfm/'
    - 'seq/'
    - 'fpca/'
    - 'vpca/'
    - 'mfpca/'
    - 'tucker/'
    - 'cp/'
    - 'plmmd/'

================================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorly as tl
from tensorly.tenalg import multi_mode_dot
from tensorly.cp_tensor import cp_to_tensor
from data_generation import data_generation


def calculate_reconstruction_error(X_valid, V):
    loss = np.linalg.norm(X_valid - X_valid @ V @ V.T, 'fro') ** 2
    return loss


def get_n_RE_matrix(n, seeds_list):
    RE_matrix = []
    seq_RE_matrix = []
    IDLFM_RE_matrix = []
    pca_RE_matrix = []
    fpca_RE_matrix = []
    mfpca_RE_matrix = []
    vpca_RE_matrix = []
    for seed in seeds_list:
        # PCA
        eigenvectors_dict = {}
        for key in keys:
            V = pd.read_csv(f"pca/n={n}, seed={seed}/{key}.csv")
            eigenvectors_dict[key] = V

        # FPCA
        fpca_eigenvectors_dict = {}
        for key in keys:
            V = pd.read_csv(f"fpca/n={n}, seed={seed}/{key}.csv")
            fpca_eigenvectors_dict[key] = V

        # MFPCA
        mfpca_eigenvectors = pd.read_csv(f"mfpca/n={n}, "
                                         f"100seeds/n={n}, seed={seed}.csv")

        # VPCA
        vpca_eigenvectors = pd.read_csv(f"vpca/n={n}/n={n}, seed={seed}.csv")
        split_size = 70
        split_num = 5
        vpca_eigenvectors_dict = {f'{keys[i]}': vpca_eigenvectors[i * split_size:(i + 1) * split_size]
                                  for i in range(split_num)}

        # PLMMD
        model_eigenvectors_dict = {}
        for key in keys:
            B = pd.read_csv(f"plmmd/n={n}, seed={seed}/"
                            f"{key}_final.csv")
            V, _ = np.linalg.qr(B)
            model_eigenvectors_dict[key] = V

        # SEQ
        seq_eigenvectors_dict = {}
        for key in keys:
            B = pd.read_csv(f"seq/n={n}, seed={seed}/"
                            f"{key}.csv")
            V, _ = np.linalg.qr(B)
            seq_eigenvectors_dict[key] = V

        # IDLFM
        IDLFM_eigenvectors_dict = {}
        for key in keys:
            B = pd.read_csv(f"idlfm/{n}_{seed}_{key}.csv")
            V, _ = np.linalg.qr(B)
            IDLFM_eigenvectors_dict[key] = V

        data_dict = data_generation(1000, seed * 3)

        pca_RE_list = []
        for key in keys:
            pca_RE = calculate_reconstruction_error(data_dict[key], eigenvectors_dict[key]) / 1000 / 70
            pca_RE_list.append(pca_RE)
        pca_RE_matrix.append(pca_RE_list)

        spca_RE_list = []
        for key in keys:
            spca_RE = calculate_reconstruction_error(data_dict[key], fpca_eigenvectors_dict[key]) / 1000 / 70
            spca_RE_list.append(spca_RE)
        fpca_RE_matrix.append(spca_RE_list)

        RE_list = []
        for key in keys:
            RE = calculate_reconstruction_error(data_dict[key], model_eigenvectors_dict[key]) / 1000 / 70
            RE_list.append(RE)
        RE_matrix.append(RE_list)

        seq_RE_list = []
        for key in keys:
            seq_RE = calculate_reconstruction_error(data_dict[key], seq_eigenvectors_dict[key]) / 1000 / 70
            seq_RE_list.append(seq_RE)
        seq_RE_matrix.append(seq_RE_list)

        IDLFM_RE_list = []
        for key in keys:
            IDLFM_RE = calculate_reconstruction_error(data_dict[key], IDLFM_eigenvectors_dict[key]) / 1000 / 70
            IDLFM_RE_list.append(IDLFM_RE)
        IDLFM_RE_matrix.append(IDLFM_RE_list)

        mfpca_RE_list = []
        for key in keys:
            mfpca_RE = calculate_reconstruction_error(data_dict[key], mfpca_eigenvectors) / 1000 / 70
            mfpca_RE_list.append(mfpca_RE)
        mfpca_RE_matrix.append(mfpca_RE_list)

        vpca_RE_list = []
        for key in keys:
            vpca_RE = calculate_reconstruction_error(data_dict[key], vpca_eigenvectors_dict[key]) / 1000 / 70
            vpca_RE_list.append(vpca_RE)
        vpca_RE_matrix.append(vpca_RE_list)

    pca_RE_matrix = np.array(pca_RE_matrix)
    fpca_RE_matrix = np.array(fpca_RE_matrix)
    mfpca_RE_matrix = np.array(mfpca_RE_matrix)
    vpca_RE_matrix = np.array(vpca_RE_matrix)
    RE_matrix = np.array(RE_matrix)
    seq_RE_matrix = np.array(seq_RE_matrix)
    IDLFM_RE_matrix = np.array(IDLFM_RE_matrix)

    return pca_RE_matrix, fpca_RE_matrix, mfpca_RE_matrix, vpca_RE_matrix, RE_matrix, seq_RE_matrix, IDLFM_RE_matrix


def get_tucker_RE_matrix(n, seeds_list):
    tucker_RE_matrix = []
    for seed in seeds_list:
        U1 = pd.read_csv(f"tucker/n={n}, seed={seed}, U1.csv")
        U2 = pd.read_csv(f"tucker/n={n}, seed={seed}, U2.csv")
        n2 = 1000
        data_dict = data_generation(n2, seed * 3)
        matrices_list_2 = [data_dict[key] for key in keys]
        transposed_matrices_2 = [matrix.T for matrix in matrices_list_2]
        tensor_2 = np.stack(transposed_matrices_2, axis=0)  # (5, 70, n2)
        tucker_error_matrices = {}
        for l in range(n2):
            matrices = [tensor_2[:, :, l]]
            matrix = np.vstack(matrices)
            Z = multi_mode_dot(matrix, [U1.T, U2.T], modes=[0, 1])
            E = matrix - multi_mode_dot(Z, [U1, U2], modes=[0, 1])
            tucker_error_matrices[l] = E
        tensor_3 = np.stack([tucker_error_matrices[l] for l in range(n2)], axis=2)
        tucker_RE_list = np.mean((tensor_2 - tensor_3) ** 2, axis=(1, 2))
        tucker_RE_matrix.append(tucker_RE_list)
    tucker_RE_matrix = np.array(tucker_RE_matrix)
    return tucker_RE_matrix


def get_cp_RE_matrix(n, seeds_list):
    cp_RE_matrix = []
    for seed in seeds_list:
        A = pd.read_csv(f"cp/n={n}, seed={seed}, A.csv")
        B = pd.read_csv(f"cp/n={n}, seed={seed}, B.csv")
        n2 = 1000
        data_dict = data_generation(n2, seed * 3)
        matrices_list_2 = [data_dict[key] for key in keys]
        transposed_matrices_2 = [matrix.T for matrix in matrices_list_2]
        tensor_2 = np.stack(transposed_matrices_2, axis=0)  # (5, 70, n2)
        cp_error_matrices = {}
        for l in range(n2):
            matrices = [tensor_2[:, :, l]]
            matrix = np.vstack(matrices)
            D = tl.tenalg.khatri_rao([B, A])
            F = D.T
            G = np.linalg.pinv(F)
            vec_Y = matrix.reshape(1, -1, order='F')
            H = vec_Y @ G
            H = np.array(H).flatten()
            new_factors = [A, B]
            cp_tensor = (H, new_factors)
            reconstructed_matrix = cp_to_tensor(cp_tensor)
            E = matrix - reconstructed_matrix
            cp_error_matrices[l] = E
        tensor_3 = np.stack([cp_error_matrices[l] for l in range(n2)], axis=2)
        cp_RE_list = np.mean((tensor_2 - tensor_3) ** 2, axis=(1, 2))
        cp_RE_matrix.append(cp_RE_list)
    cp_RE_matrix = np.array(cp_RE_matrix)
    return cp_RE_matrix


def get_slide_RE_matrix(n, seeds_list):
    slide_RE_matrix = []
    for seed in seeds_list:
        B = pd.read_csv(f"slide/n={n}, seed={seed}, V.csv")
        n2 = 1000
        data_dict = data_generation(n2, seed * 3)
        test_data = np.hstack(list(data_dict.values()))
        V, _ = np.linalg.qr(B)
        error_matrix = test_data - test_data @ V @ V.T
        error_matrix = np.array(error_matrix)
        slide_RE_list = [
            np.mean(error_matrix[:, i:i + 70] ** 2)
            for i in range(0, error_matrix.shape[1], 70)
        ]
        slide_RE_list = np.array(slide_RE_list)
        slide_RE_matrix.append(slide_RE_list)
    return np.array(slide_RE_matrix)


path = ""
keys = ['data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW']
pc_numbers = [7, 7, 6, 6, 8]
best_params = [364986, 555817, 511958, 550696, 612431]
tru_eigenvectors_dict = {}
for key in keys:
    B = pd.read_csv(
        f"eigenvectors/{key}.csv", header=None)
    tru_eigenvectors_dict[key] = B.values

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

# ==== boxplot for n=200 ====
n = 200
pca_RE_matrix, spca_RE_matrix, mfpca_RE_matrix, vpca_RE_matrix, RE_matrix, seq_RE_matrix, IDLFM_RE_matrix = get_n_RE_matrix(
    n, seeds_list)
tucker_RE_matrix = get_tucker_RE_matrix(n, seeds_list)
cp_RE_matrix = get_cp_RE_matrix(n, seeds_list)
slide_RE_matrix = get_slide_RE_matrix(n, seeds_list)

for index in range(5):
    # boxplot of one station
    df1 = pd.DataFrame({
        'Reconstruction Error': np.concatenate([spca_RE_matrix[:, index],
                                                mfpca_RE_matrix[:, index],
                                                vpca_RE_matrix[:, index],
                                                tucker_RE_matrix[:, index],
                                                cp_RE_matrix[:, index],
                                                slide_RE_matrix[:, index],
                                                seq_RE_matrix[:, index],
                                                IDLFM_RE_matrix[:, index],
                                                RE_matrix[:, index]]),
        'Method': (['FPCA'] * len(spca_RE_matrix[:, index]) +
                   ['MFPCA'] * len(mfpca_RE_matrix[:, index]) +
                   ['VPCA'] * len(vpca_RE_matrix[:, index]) +
                   ['Tucker'] * len(tucker_RE_matrix[:, index]) +
                   ['CP'] * len(cp_RE_matrix[:, index]) +
                   ['SLIDE'] * len(slide_RE_matrix[:, index]) +
                   ['SEQ'] * len(seq_RE_matrix[:, index]) +
                   ['IDLFM'] * len(IDLFM_RE_matrix[:, index]) +
                   [rf'$PLM^{2}D$'] * len(RE_matrix[:, index]))
    })

    plt.figure(figsize=(5, 5))
    meanprops = {"marker": "o",
                 "markerfacecolor": "white",
                 "markeredgecolor": "black",
                 "markersize": 5}

    # Create vertical boxplot (change orient='h' to orient='v')
    sns.boxplot(x='Method', y='Reconstruction Error', data=df1, orient='v', palette='RdYlBu',
                showmeans=True, meanprops=meanprops, showfliers=False)

    plt.ylabel('MSE', fontsize=18)
    plt.xlabel('')
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    title = f'Reconstruction Error Boxplot of {keys[index][5:]} when n={n}'

    # Save the plot
    plt.savefig(path + title + '.png', bbox_inches='tight', dpi=300)
    plt.show()


# ==== boxplot over n ====
n_list = [50, 100, 200, 300, 400]
RE_matrices = {
    rf'$FPCA$': {},
    rf'$MFPCA$': {},
    rf'$VPCA$': {},
    rf'$Tucker$': {},
    rf'$CP$': {},
    rf'$SLIDE$': {},
    rf'$SEQ$': {},
    rf'$IDLFM$': {},
    rf'$PLM^{2}D$': {}
}
for n in n_list:
    pca_RE_matrix, pca_hier_RE_matrix, spca_RE_matrix, seq_RE_matrix, mfpca_RE_matrix, vpca_RE_matrix, RE_matrix, seq_RE_matrix, IDLFM_RE_matrix = get_n_RE_matrix(n, seeds_list)
    tucker_RE_matrix = get_tucker_RE_matrix(n, seeds_list)
    cp_RE_matrix = get_cp_RE_matrix(n, seeds_list)
    slide_RE_matrix = get_slide_RE_matrix(n, seeds_list)
    RE_matrices[rf'$FPCA$'][n] = spca_RE_matrix
    RE_matrices[rf'$MFPCA$'][n] = mfpca_RE_matrix
    RE_matrices[rf'$VPCA$'][n] = vpca_RE_matrix
    RE_matrices[rf'$Tucker$'][n] = tucker_RE_matrix
    RE_matrices[rf'$CP$'][n] = cp_RE_matrix
    RE_matrices[rf'$SLIDE$'][n] = slide_RE_matrix
    RE_matrices[rf'$SEQ$'][n] = seq_RE_matrix
    RE_matrices[rf'$IDLFM$'][n] = seq_RE_matrix
    RE_matrices[rf'$PLM^{2}D$'][n] = RE_matrix

data1 = []
for method, values in RE_matrices.items():
    for n, matrix in values.items():
        for row in matrix:
            for (idx, value) in enumerate(row):
                data1.append({'n': n, 'RE': value, 'Method': method, 'Station': keys[idx][5:]})
df1 = pd.DataFrame(data1)
for key in keys:
    station_data = df1[df1['Station'] == key[5:]]
    plt.figure(figsize=(20, 6))
    meanprops = {"marker": "o",
                 "markerfacecolor": "white",
                 "markeredgecolor": "black",
                 "markersize": 5}
    flierprops = {"marker": "o",
                  "markerfacecolor": "gray",
                  "markeredgecolor": "none",
                  "markersize": 3}
    ax = sns.boxplot(x='n', y='RE', hue='Method', data=station_data,
                     palette='RdYlBu', showmeans=True, meanprops=meanprops,
                     showfliers=False, flierprops=flierprops
                     )
    title = f'Reconstruction Error for different methods over n at Station {key[5:]}'

    plt.grid(True, ls="--", linewidth=0.6, alpha=0.7, axis='y')

    plt.axvline(x=0.5, color='grey', linestyle='-.')
    plt.axvline(x=1.5, color='grey', linestyle='-.')
    plt.axvline(x=2.5, color='grey', linestyle='-.')
    plt.axvline(x=3.5, color='grey', linestyle='-.')
    plt.axvline(x=4.5, color='grey', linestyle='-.')
    plt.xlabel('Sample Size n', fontsize=18)
    plt.ylabel('Reconstruction Error', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title='Method', title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
    plt.tight_layout()
    plt.savefig(path + title + '.png', dpi=300)

    plt.show()

new_RE_matrices = {
    method: {
        sample_size: np.sum(error_matrix, axis=1, keepdims=True)
        for sample_size, error_matrix in matrix.items()
    }
    for method, matrix in RE_matrices.items()
}
data2 = []
for method, values in new_RE_matrices.items():
    for n, matrix in values.items():
        for row in matrix:
            for (idx, value) in enumerate(row):
                data2.append({'n': n, 'RE': value, 'Method': method})
df2 = pd.DataFrame(data2)
station_data = df2
n_values = [50, 100, 200, 300, 400]
method_values = df2['Method'].unique()
xtick_labels = np.tile(method_values, 5)

plt.figure(figsize=(20, 10))
meanprops = {"marker": "o",
             "markerfacecolor": "white",
             "markeredgecolor": "black",
             "markersize": 5}
flierprops = {"marker": "o",
              "markerfacecolor": "gray",
              "markeredgecolor": "none",
              "markersize": 3}
ax = sns.boxplot(x='n', y='RE', hue='Method', data=station_data,
                 palette='RdYlBu', showmeans=True, meanprops=meanprops,
                 showfliers=False, flierprops=flierprops
                 )
title = 'Total reconstruction error for different methods over n'

ax.set_xticklabels([])

box_positions = []
for line in ax.lines:
    if line.get_marker() == 'o':
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        for x, y in zip(xdata, ydata):
            box_positions.append(x)

plt.xticks(ticks=box_positions, labels=xtick_labels, rotation=90, fontsize=12)

plt.grid(True, ls="--", linewidth=0.6, alpha=0.7, axis='y')
plt.legend([], [], frameon=False)
plt.axvline(x=0.5, color='grey', linestyle='-.')
plt.axvline(x=1.5, color='grey', linestyle='-.')
plt.axvline(x=2.5, color='grey', linestyle='-.')
plt.axvline(x=3.5, color='grey', linestyle='-.')
plt.text(0, 15.1, 'n=50', fontsize=18, color='black', ha='center', va='center')
plt.text(1, 15.1, 'n=100', fontsize=18, color='black', ha='center', va='center')
plt.text(2, 15.1, 'n=200', fontsize=18, color='black', ha='center', va='center')
plt.text(3, 15.1, 'n=300', fontsize=18, color='black', ha='center', va='center')
plt.text(4, 15.1, 'n=400', fontsize=18, color='black', ha='center', va='center')
plt.xlabel('Training Data Size n', fontsize=18)
plt.ylabel('MSE', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

plt.savefig(path + "C:/Users/MRY/Desktop/Total reconstruction error for different methods over n.png", dpi=300)
plt.show()
