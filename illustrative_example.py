import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib as mpl


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

    #     if not converged:
    #         print("Did not converge within the maximum number of iterations.")

    #     for i in range(pc_number):
    #         b_dict[f'b{i + 1}'] = b_dict[f'b{i + 1}'] / np.linalg.norm(b_dict[f'b{i + 1}'])
    #     v_values = [b_dict[f'b{i + 1}'] for i in range(pc_number)]
    #     V = np.column_stack(v_values)

    return B


# load data
data = np.load("MTR.npy")
start_time = dt.datetime(2023, 1, 1, 6, 30)
end_time = dt.datetime(2023, 1, 2, 0, 0)
time_seg = 15
Time_name = pd.date_range(start=start_time, end=end_time, freq=str(time_seg) + 'T', inclusive='left')
Time_name = Time_name.strftime('%H:%M')
keys = ['data_CEN', 'data_ADM', 'data_SHW', 'data_WAC',
        'data_CAB', 'data_TIH', 'data_FOH', 'data_NOP',
        'data_QUB', 'data_TAK', 'data_SWH', 'data_SKW',
        'data_HFC', 'data_CHW', 'data_SYP', 'data_HKU',
        'data_KET'
        ]
path = ""

k_list = [9, 5, 8]
a_list = [62, 27, 31]
alpha_list = [67646, 1945, 324114]
colors = ['#d73027', '#4575b4', '#fdae61']
stations = ['HKU', 'CAB', 'QUB']


# subfigure 1 (top two eigenfunctions of HKU, CAB and QUB)
for j, k in enumerate(k_list):
    pca = PCA(n_components=k)
    data_CEN = data[:, :, 0, a_list[j], :].sum(axis=1)
    B = smoothPCA(data_CEN, k, alpha_list[j])
    data_CEN_pca, _ = np.linalg.qr(B)
    title = f'{stations[j]}'

    plt.figure(figsize=(9, 6))
    plot_handles = []
    for i in range(2):
        vec = data_CEN_pca[:, i]
        if j == 1 and i == 1:
            vec = - vec
        handle, = plt.plot(vec, label=f'eigenvector {i + 1}', color=colors[i])
        plot_handles.append(handle)
    plt.title(title, fontsize=24, y=1.025)
    plt.xlabel('Time', fontsize=20)
    plt.yticks(fontsize=18)
    plt.xticks(np.arange(0, len(Time_name), 5), Time_name[::5], fontsize=18, rotation=45)
    plt.ylim(-0.45, 0.45)
    plt.tight_layout()
    plt.savefig(path + title + '.png', dpi=300)
    plt.show()

plt.figure(figsize=(5, 3))
plt.legend(plot_handles, ['1st eigenfunction', '2nd eigenfunction'], loc='center', fontsize=16, frameon=False)
plt.axis('off')
plt.tight_layout()
plt.savefig(path + 'legend.png', dpi=300)
plt.show()

# subfigure 2 (heatmaps of Pearson correlation coefficients between PC scores of HKU, CAB and QUB)

pca = PCA(n_components=k_list[0])
data_HKU = data[:, :, 0, a_list[0], :].sum(axis=1)
data_HKU_pca = pca.fit_transform(data_HKU)

pca = PCA(n_components=k_list[1])
data_CAB = data[:, :, 0, a_list[1], :].sum(axis=1)
data_CAB_pca = pca.fit_transform(data_CAB)

pca = PCA(n_components=k_list[2])
data_QUB = data[:, :, 0, a_list[2], :].sum(axis=1)
data_QUB_pca = pca.fit_transform(data_QUB)

k1 = 9
k2 = 5
k3 = 8
corr_matrix = np.zeros((k1, k2))
for i in range(k1):
    for j in range(k2):
        r, _ = pearsonr(data_HKU_pca[:, i], data_CAB_pca[:, j])
        corr_matrix[i, j] = abs(r)

df_corr = pd.DataFrame(corr_matrix,
                       index=[f'{i + 1}' for i in range(k1)],
                       columns=[f'{j + 1}' for j in range(k2)])

plt.figure(figsize=(6, 6))
sns.heatmap(df_corr, annot=True, cmap='Blues', fmt=".2f", square=True,
            cbar=False, annot_kws={"size": 12})
plt.xlabel("CAB", fontsize=14)
plt.ylabel("HKU", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14, rotation=0)
plt.tight_layout()
plt.savefig(path + 'correlation_1.png', dpi=300)
plt.show()

corr_matrix = np.zeros((k1, k3))

for i in range(k1):
    for j in range(k3):
        r, _ = pearsonr(data_HKU_pca[:, i], data_QUB_pca[:, j])
        corr_matrix[i, j] = abs(r)

df_corr = pd.DataFrame(corr_matrix,
                       index=[f'{i + 1}' for i in range(k1)],
                       columns=[f'{j + 1}' for j in range(k3)])

plt.figure(figsize=(6, 6))
sns.heatmap(df_corr, annot=True, cmap='Blues', fmt=".2f", square=True,
            cbar=False, annot_kws={"size": 12})
plt.xlabel("QUB", fontsize=14)
plt.ylabel("HKU", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14, rotation=0)
plt.tight_layout()
plt.savefig(path + 'correlation_2.png', dpi=300)
plt.show()

corr_matrix = np.zeros((k2, k3))

for i in range(k2):
    for j in range(k3):
        r, _ = pearsonr(data_CAB_pca[:, i], data_QUB_pca[:, j])
        corr_matrix[i, j] = abs(r)

df_corr = pd.DataFrame(corr_matrix,
                       index=[f'{i + 1}' for i in range(k2)],
                       columns=[f'{j + 1}' for j in range(k3)])

plt.figure(figsize=(6, 6))
sns.heatmap(df_corr, annot=True, cmap='Blues', fmt=".2f", square=True,
            cbar=False, annot_kws={"size": 12})
plt.xlabel("QUB", fontsize=14)
plt.ylabel("CAB", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14, rotation=0)
plt.tight_layout()
plt.savefig(path + 'correlation_3.png', dpi=300)
plt.show()

# colobar
cmap = plt.get_cmap('Blues')
norm = mpl.colors.Normalize(vmin=0, vmax=1)
fig, ax = plt.subplots(figsize=(2, 4.0))
fig.subplots_adjust(left=0.5, right=0.6)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
cb.set_label('Correlation', fontsize=14, labelpad=15)
cb.ax.tick_params(labelsize=12)
plt.savefig(path + 'colorbar.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
