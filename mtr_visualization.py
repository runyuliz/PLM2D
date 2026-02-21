"""
================================================================================
This script visualizes the results obtained from the PLMMD algorithm.
It generates two types of plots (Figures 5-6 in the paper):
1. Loading Vectors: Visualizing the temporal patterns of each cluster.
2. Passenger Flow Profiles: Visualizing how these patterns perturb the mean flow.

--------------------------------------------------------------------------------
REQUIRED INPUTS:
--------------------------------------------------------------------------------
1.  DATA FILES:
    - 'MTR.npy': The original 3D numpy array (Days x Stations x TimePoints).
    - 'station_indices.xlsx': Station naming and indexing mapping.
    - 'structure_5stations_original.csv': The final clustering structure (only five selected stations).

2.  ALGORITHM OUTPUTS:
    - './B/{key}_final.csv': Loading matrices (B) saved for each station after
      running the PLMMD algorithm.

================================================================================
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns

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

validation_datas = {}
for key, value in datas.items():
    validation_data = value[validation_days, :]
    validation_datas[key] = validation_data

test_datas = {}
for key, value in datas.items():
    test_data = value[test_days, :]
    test_datas[key] = test_data

keys_to_save = ['data_WAC', 'data_TIH', 'data_TAK', 'data_SKW', 'data_CHW']
keys_to_remove = [key for key in test_datas if key not in keys_to_save]
for key in keys_to_remove:
    del test_datas[key]

mean_train_datas = {}
for key, value in train_datas.items():
    mean = np.average(value, axis=0)
    mean_train_datas[key] = mean

start_time = dt.datetime(2023, 1, 1, 6, 30)
end_time = dt.datetime(2023, 1, 2, 0, 0)
time_seg = 15
Time_name = pd.date_range(start=start_time, end=end_time, freq=str(time_seg) + 'T', inclusive='left')
Time_name = Time_name.strftime('%H:%M')

# get model eigenvectors after PLMMD
model_eigenvectors_dict = {}
for key in keys:
    model_B = pd.read_csv(path + f"B/{key}_final.csv")
    model_eigenvectors, _ = np.linalg.qr(model_B)
    model_eigenvectors_dict[key] = model_eigenvectors

# get final structure after PLMMD (5 selected stations)
cluster_indices = pd.read_csv("structure_5stations_original.csv")
cluster_indices['indices'] = cluster_indices['indices'].str.strip('[]')
cluster_eigenvectors_dict = {}
for i in cluster_indices.index:
    cluster_eigenvectors_dict[i] = {}
    indices_list = cluster_indices.at[i, 'indices'].replace("'", "").split(", ")
    for indices in indices_list:
        matrix_index, col_index = indices.split('_')
        matrix_index = int(matrix_index) - 1
        col_index = int(col_index) - 1
        key = keys[matrix_index]
        cluster_eigenvectors_dict[i][f'{key}_{col_index + 1}'] = model_eigenvectors_dict[key][:, col_index]


save_path = 'cluster/'
os.makedirs(save_path, exist_ok=True)

# plot loading vectors
for outer_key, inner_dict in cluster_eigenvectors_dict.items():
    title = f"Cluster-{outer_key + 1}"

    all_values = np.concatenate(list(inner_dict.values()))
    y_min, y_max = all_values.min(), all_values.max()

    for inner_key, value in inner_dict.items():
        plt.figure(figsize=(10, 5))
        color = sns.color_palette(['#d15c6b', '#f5cf36', '#8fb943', '#78b9d2', '#8386a8'])[
            list(inner_dict.keys()).index(inner_key)]
        sns.lineplot(x=np.arange(len(value)), y=value, color=color, linewidth=2.5)

        plt.xlabel('Time', fontsize=16)
        plt.xticks(np.arange(0, len(Time_name), 5), Time_name[::5], fontsize=14, rotation=45)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--')

        ax = plt.gca()
        ax.set_ylim(y_min - 0.1, y_max + 0.1)

        plt.axhline(0, color='gray', linestyle='--', linewidth=2.5)
        plt.tight_layout()

        plot_filename = f"Cluster_{outer_key + 1}_{inner_key[5:]}_1.png"
        plt.savefig(save_path + plot_filename, dpi=300)
        plt.show()

# plot passenger flow profile
for outer_key, inner_dict in cluster_eigenvectors_dict.items():
    title = f"Cluster-{outer_key + 1}"
    palette = sns.color_palette(['#d15c6b', '#f5cf36', '#8fb943', '#78b9d2', '#8386a8'])

    for idx, (inner_key, value) in enumerate(inner_dict.items()):
        plt.figure(figsize=(10, 5))

        color = palette[idx % len(palette)]
        coefficient = 1000

        plt.plot(mean_train_datas[inner_key[0:8]], color=color, linewidth=2.5, label=f'{inner_key[5:8]}')
        plt.plot(coefficient * value + mean_train_datas[inner_key[0:8]],
                 color='#8c8c8c', marker='+', markersize=10, markeredgewidth=2.2,
                 linewidth=2, markevery=(1, 3), zorder=3)
        plt.plot(-coefficient * value + mean_train_datas[inner_key[0:8]],
                 color='#8c8c8c', linestyle='--', linewidth=2.5)

        plt.xlabel('Time', fontsize=16)
        plt.xticks(np.arange(0, len(Time_name), 5), Time_name[::5], rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--')
        plt.tight_layout()

        plot_filename = f"Cluster_{outer_key + 1}_{inner_key[5:]}_2.png"
        plt.savefig(save_path + plot_filename, dpi=300)
        plt.show()
