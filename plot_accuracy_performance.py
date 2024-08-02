# Plotting performance CNNs comparison

# Angel Canelo 2024.08.02
######### IMPORT ##########
import glob
import pandas as pd
from pymatreader import read_mat
import matplotlib.pyplot as plt
import seaborn as sns
###########################
rew_dic = {"Test_accuracy": [], "Epoch": [], "Trial": [], "Net": []}
rew_dic2 = {"Test_accuracy": [], "Epoch": [], "Trial": [], "Net": []}
standard_dataset = 0   # 0 -> Moving Patterns dataset; 1 -> COIL100
if standard_dataset == 0:
    file_directory = "..\performance_mat\\*_Moving_Pattern_perf_loop*.mat"
elif standard_dataset == 1:
    file_directory = "..\performance_mat\\*_COIL100_perf_loop*"
file_list = glob.glob(file_directory)
ii = 0; iii=0
for j, file_path in enumerate(file_list):
    to_file = read_mat(file_path)
    if '244X324' in file_path:
        for i in range(len(to_file["hist_testacc"])):
            rew_dic["Test_accuracy"].append(to_file["hist_acc"][i])
            rew_dic["Epoch"].append(i)
            rew_dic["Trial"].append(ii)
            if 'FlyVisNet' in file_path:
                rew_dic["Net"].append("FlyVisNet")
            elif 'Dronet' in file_path:
                rew_dic["Net"].append("Dronet")
            elif 'MobileNetV2' in file_path:
                rew_dic["Net"].append("MobileNetV2")
            elif 'Random_init' in file_path:
                rew_dic["Net"].append("Random_init")
        ii += 1
    elif '20X40' in file_path:
        for i in range(len(to_file["hist_testacc"])):
            rew_dic2["Test_accuracy"].append(to_file["hist_acc"][i])
            rew_dic2["Epoch"].append(i)
            rew_dic2["Trial"].append(iii)
            if 'FlyVisNet' in file_path:
                rew_dic2["Net"].append("FlyVisNet")
            elif 'Dronet' in file_path:
                rew_dic2["Net"].append("Dronet")
            elif 'MobileNetV2' in file_path:
                rew_dic2["Net"].append("MobileNetV2")
            elif 'Random_init' in file_path:
                rew_dic2["Net"].append("Random_init")
        iii += 1

df_r = pd.DataFrame(data=rew_dic, columns=["Epoch", "Test_accuracy", "Trial", "Net"])
df_r2 = pd.DataFrame(data=rew_dic2, columns=["Epoch", "Test_accuracy", "Trial", "Net"])
max_values = df_r.groupby(['Trial', 'Net']).max()

print('Accuracies for 244x324')
print(max_values)
print(max_values.groupby(['Net']).mean())
print(max_values.groupby(['Net']).std())
print(max_values.groupby(['Net']).max())

max_values = df_r2.groupby(['Trial', 'Net']).max()

print('Accuracies for 20x40')
print(max_values)
print(max_values.groupby(['Net']).mean())
print(max_values.groupby(['Net']).std())
print(max_values.groupby(['Net']).max())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
sns.lineplot(x="Epoch", y="Test_accuracy", data=df_r, hue="Net", hue_order=["FlyVisNet", "Dronet", "MobileNetV2", "Random_init"],
             errorbar=('ci', 95), palette=['blue', 'red', 'green', 'magenta'], ax=ax1)
ax1.grid(True)
ax1.set_title('Performance on pattern dataset (ci 95) \n (3000/300 frames train/test)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_xlabel('Epoch (#)')
ax1.set_xlim([0, 100])
ax1.legend(loc='lower right')

sns.lineplot(x="Epoch", y="Test_accuracy", data=df_r2, hue="Net", hue_order=["FlyVisNet", "MobileNetV2", "Random_init"],
             errorbar=('ci', 95), palette=['blue', 'green', 'magenta'], ax=ax2)
ax2.grid(True)
ax2.set_ylabel('Accuracy (%)')
ax2.set_xlabel('Epoch (#)')
ax2.legend(loc='lower right')
ax2.set_xlim([0, 100])
plt.show()