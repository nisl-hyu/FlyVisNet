# Plotting pruning accuracy

# Angel Canelo 2024.08.02

from pymatreader import read_mat
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

prune = 1 # 0 -> Pruning by layers; 1 -> Global pruning
if prune == 0:
    layer_names = ['R16_kernel', 'L1_kernel', 'L2_kernel', 'L3_kernel', 'Mi1_kernel', 'Tm3_kernel', 'C3_kernel', 'Mi4_kernel', 'Tm1_kernel',
                   'Tm2_kernel', 'Tm4_kernel', 'Mi9_kernel', 'Tm9_kernel', 'CT1_kernel', 'TmY9_kernel', 'TmY4_kernel', 'TmY5_kernel',
                   'T2_kernel', 'T3_kernel', 'T4_kernel', 'T5_kernel', 'Li_kernel', 'LPi_kernel', 'LC11_kernel', 'LC15_kernel', 'LPLC2_kernel']

    rew_dic = {"Test_accuracy": [], "layer_names": []}

    for j in range(1, 11):
        rew_dic["Test_accuracy"].append(0.9853)
        rew_dic["layer_names"].append('Original')
        rew_dic["Test_accuracy"].append(0.9766)
        rew_dic["layer_names"].append(layer_names[0])
        for i in range(1,len(layer_names)):
            data = read_mat(f"../performance_mat/PRUNING/FlyVisNet_244X324_Moving_Pattern_perf_loop_{layer_names[i]}_{j}.mat")
            rew_dic["Test_accuracy"].append(data['topmax_test'])
            rew_dic["layer_names"].append(layer_names[i])

    df_r = pd.DataFrame(data=rew_dic, columns=["Test_accuracy", "layer_names"])
    mean_accuracy = df_r.groupby('layer_names')['Test_accuracy'].mean().reset_index()
    mean_accuracy_sorted = mean_accuracy.sort_values(by="Test_accuracy", ascending=False)
    df_r_sorted = df_r.set_index('layer_names').loc[mean_accuracy_sorted['layer_names']].reset_index()
elif prune == 1:
    prun_glob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    rew_dic = {"Test_accuracy": [], "layer_names": []}

    for j in range(0, 10):
        rew_dic["Test_accuracy"].append(0.9853)
        rew_dic["layer_names"].append('0.0')
        for i in range(0, len(prun_glob)):
            data = read_mat(f"../performance_mat/PRUNING/FlyVisNet_pruned_global_perf_{prun_glob[i]}_{j}.mat")
            rew_dic["Test_accuracy"].append(data['topmax_test'])
            rew_dic["layer_names"].append(f"{prun_glob[i]}")

    df_r = pd.DataFrame(data=rew_dic, columns=["Test_accuracy", "layer_names"])
    mean_accuracy = df_r.groupby('layer_names')['Test_accuracy'].mean().reset_index()
    mean_accuracy_sorted = mean_accuracy.sort_values(by="Test_accuracy", ascending=False)
    df_r_sorted = df_r.set_index('layer_names').loc[mean_accuracy_sorted['layer_names']].reset_index()

fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
sns.lineplot(x="layer_names", y="Test_accuracy", data=df_r_sorted, errorbar=('ci', 95), err_style='bars', ax=ax1)
ax1.grid(True, linestyle=':')
# ax1.set_ylim(0.85, 1)
plt.xticks(rotation=45)
plt.show()