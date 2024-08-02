# FlyVisNet activity

# Angel Canelo 2024.08.02

###### import ######################
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from FlyVisNetH_model import FlyVisNetH
##################################
tf.keras.backend.clear_session()
##################################
def getLayerIndexByName(model, layername):
  for idx, layer in enumerate(model.layers):
    if layer.name == layername:
      return idx
##################################
###### Load model ######
HEIGHT = 244
WIDTH = 324
n_out = 3
cnn = FlyVisNetH()
cnn_model = cnn.FlyVisNet_model(HEIGHT, WIDTH, n_out)
# Load pre-trained model
cnn_model.load_weights(f"../WEIGHTS/FlyVisNet_weights_244X324_Moving_Pattern_0.h5")
# cnn_model.summary()
tmax = int(344/2)+13  #142
#########################################
layer_names = ['T4', 'T5', 'LC11', 'LC15', 'LPLC2']; layer_outputs = []
excluded_layer_names = ['LPLC2', 'input_1', 'max_pooling2d', 'concatenate', 'lambda', 'conv2d', 'dense', 'kernel', 'flatten', 'classification']
for ele in layer_names:
    ind_layer = getLayerIndexByName(cnn_model, ele)
    if ind_layer is None:
        continue
    else:
        layer_outputs.append(cnn_model.layers[ind_layer].output)
activation_model = tf.keras.models.Model(inputs=cnn_model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
###### Generate stimulus ######
dark_base = 0
barfr = dark_base*np.ones((244, 324, 1), np.uint8)
barfr2 = dark_base*np.ones((244, 324, 1), np.uint8)
loomfr = dark_base*np.ones((244, 324, 1), np.uint8)
plt.figure()
# 2D activation lists init
LC11_act2D = []; LC15_act2D = []; T4_act2D = []; T5_act2D = []; T4ND_act2D = []; T5ND_act2D = []; lamedul_act2D =[]; conv_weights =[]
std_2d = 8
std2_2d = 5
lamedul_names = []; kernel_names = []
for n in range(2):
    INPUT1 = []; INPUT2 = []
    if n==0:
        height = 150
        width = 5
        t_trace = np.linspace(0,10,tmax)
        vel = 2
        lc11_trace = []; lc15_trace = []; T4_trace = []; T5_trace = []; T4_trace_ND = []; T5_trace_ND = []; lamedul=[]
        LPLC2_trace = []; classif_trace = []
        gauss_std = 1.25
    elif n==1:
        height = 5
        width = 5
        vel = 2
        lc11_trace = []; lc15_trace = []; T4_trace = []; T5_trace = []; T4_trace_ND = []; T5_trace_ND = []
        LPLC2_trace = []; classif_trace = []
        gauss_std = 1.25
    nn = 0
    width2 = 2
    for k in range(tmax):
        if k<=10:
            INPUT1.append(barfr)
            INPUT2.append(barfr2)
        elif k>=172:
            INPUT1.append(barfr)
            INPUT2.append(barfr2)
        else:
            if min([width, width2]) < 5:
                INPUT1.append(cv2.rectangle(barfr,(0,120-int(height/2)),(min([width, width2]),120+int(height/2)),(127,127,127),cv2.FILLED))
                INPUT2.append(cv2.rectangle(barfr2, (324 * vel, 120 - int(height / 2)), (324-min([width, width2]), 120 + int(height / 2)),
                              (127, 127, 127), cv2.FILLED))
            else:
                INPUT1.append(cv2.rectangle(barfr,((k-11)*vel,120-int(height/2)),(width+(k-11)*vel,120+int(height/2)),(127,127,127),cv2.FILLED))
                INPUT2.append(cv2.rectangle(barfr2, (324-(k-11)*vel, 120 - int(height / 2)), (324-width-(k-11)*vel, 120 + int(height / 2)),
                              (127, 127, 127), cv2.FILLED))
            width2 = width2 + vel
        barfr = dark_base*np.ones((244, 324, 1), np.uint8)
        barfr2 = dark_base*np.ones((244, 324, 1), np.uint8)
    frames = np.expand_dims(np.array(INPUT1), axis=-1)
    frames2 = np.expand_dims(np.array(INPUT2), axis=-1)
    for layer_activation in cnn_model.layers:
        if 'LC11' == layer_activation.name:
            get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
            layer_activations = get_activations([frames])[0]
            activation = np.squeeze(layer_activations)
            lc11_gauss = []
            for jj in range(tmax):
                lc11_trace.append(np.mean(activation[jj,:,:]))
                lc11_gauss.append(activation[jj, :, :])
            LC11_act2D.append(np.mean(np.array(lc11_gauss), axis=0))
        elif 'LC15' == layer_activation.name:
            get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
            layer_activations = get_activations([frames])[0]
            activation = np.squeeze(layer_activations)
            lc15_gauss = []
            for jj in range(tmax):
                lc15_trace.append(np.mean(activation[jj,:,:]))
                lc15_gauss.append(activation[jj, :, :])
            LC15_act2D.append(np.mean(np.array(lc15_gauss), axis=0))
        elif 'LPLC2' == layer_activation.name:
            get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
            layer_activations = get_activations([frames])[0]
            activation = np.squeeze(layer_activations)
            for jj in range(tmax):
                LPLC2_trace.append(np.mean(activation[jj,:,:]))
        elif 'classification' == layer_activation.name:
            get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
            layer_activations = get_activations([frames])[0]
            activation = np.squeeze(layer_activations)
            for jj in range(tmax):
                classif_trace.append(activation[jj, :])
        elif 'T4' == layer_activation.name:
            get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
            layer_activations = get_activations([frames])[0]
            activation = np.squeeze(layer_activations)
            T4_gauss = []
            for jj in range(tmax):
                T4_trace.append(np.mean(activation[jj,:,:]))
                T4_gauss.append(activation[jj, :, :])
            T4_act2D.append(np.mean(np.array(T4_gauss), axis=0))
        elif 'T5' == layer_activation.name:
            get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
            layer_activations = get_activations([frames])[0]
            activation = np.squeeze(layer_activations)
            T5_gauss = []
            for jj in range(tmax):
                T5_trace.append(np.mean(activation[jj,:,:]))
                T5_gauss.append(activation[jj, :, :])
            T5_act2D.append(np.mean(np.array(T5_gauss), axis=0))
        elif all(excluded_name not in layer_activation.name for excluded_name in excluded_layer_names) and n==0:
            lamedul_names.append(layer_activation.name)
            get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
            layer_activations = get_activations([frames])[0]
            activation = np.squeeze(layer_activations)
            lamedul_gauss = []
            for jj in range(tmax):
                lamedul.append(np.mean(activation[jj,:,:]))
                lamedul_gauss.append(activation[jj, :, :])
            lamedul_act2D.append(np.mean(np.array(lamedul_gauss), axis=0))
            if 'R16' in layer_activation.name:
                kernel_names.append(layer_activation.name)
                conv_weights.append(layer_activation.get_weights()[0])
        elif 'kernel' in layer_activation.name and n==0:
            kernel_names.append(layer_activation.name)
            conv_weights.append(layer_activation.get_weights()[0])
    for layer_activation in cnn_model.layers:
        if 'T4' == layer_activation.name:
            get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
            layer_activations = get_activations([frames2])[0]
            activation = np.squeeze(layer_activations)
            T4_gauss = []
            for jj in range(tmax):
                T4_trace_ND.append(np.mean(activation[jj,:,:]))
                T4_gauss.append(activation[jj, :, :])
            T4ND_act2D.append(np.mean(np.array(T4_gauss), axis=0))
        if 'T5' == layer_activation.name:
            get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
            layer_activations = get_activations([frames2])[0]
            activation = np.squeeze(layer_activations)
            T5_gauss = []
            for jj in range(tmax):
                T5_trace_ND.append(np.mean(activation[jj,:,:]))
                T5_gauss.append(activation[jj, :, :])
            T5ND_act2D.append(np.mean(np.array(T5_gauss), axis=0))
#########################################
######## Plotting results ###############
    if n==0:
        T4_b = np.array(T4_trace)
        T5_b = np.array(T5_trace)
        T4_b_ND = np.array(T4_trace_ND)
        T5_b_ND = np.array(T5_trace_ND)
        plt.subplot(2, 3, 1)
        plt.plot(t_trace, lc15_trace, 'b')
        plt.plot(t_trace, lc11_trace, 'r')
        plt.plot(t_trace, LPLC2_trace, 'g')
        plt.legend(['LC15', 'LC11'])
        plt.title('Moving rectangle')
        plt.ylabel('Normalized response (a.u.)')

        plt.subplot(2, 3, 4)
        f_elements = [classif_trace[i][0] for i in range(185)]
        plt.plot(t_trace, f_elements, 'b')
        s_elements = [classif_trace[i][1] for i in range(185)]
        plt.plot(t_trace, s_elements, 'r')
        t_elements = [classif_trace[i][2] for i in range(185)]
        plt.plot(t_trace, t_elements, 'g')

        plt.legend(['Loom', 'Bar', 'Spot'])

        Rmax_rect = np.max(lc15_trace/np.max(lc15_trace))
        Rmin_rect = np.max(lc11_trace / np.max(lc15_trace))
    elif n == 1:
        plt.subplot(2, 3, 2)
        plt.plot(t_trace, lc15_trace, 'b')
        plt.plot(t_trace, lc11_trace, 'r')
        plt.plot(t_trace, LPLC2_trace, 'g')
        plt.legend(['LC15', 'LC11'])
        plt.title('Moving square')
        plt.ylabel('Normalized response (a.u.)')

        plt.subplot(2, 3, 5)
        f_elements = [classif_trace[i][0] for i in range(185)]
        plt.plot(t_trace, f_elements, 'b')
        s_elements = [classif_trace[i][1] for i in range(185)]
        plt.plot(t_trace, s_elements, 'r')
        t_elements = [classif_trace[i][2] for i in range(185)]
        plt.plot(t_trace, t_elements, 'g')

        plt.legend(['Loom', 'Bar', 'Spot'])

        Rmax_square = np.max(lc11_trace/np.max(lc11_trace))
        Rmin_square = np.max(lc15_trace/np.max(lc11_trace))

lc4_trace = []; lplc2_trace = []; lc15_trace_loom = []; lc11_trace_loom = []; classif_trace_loom = []
gauss_std = 1.25
INPUT1 = []
t_trace2 = np.linspace(0, 5, tmax)
def sigmoid(x):
    # return 1 / (1 + np.exp((x - 2.5) * 20))   # Shrinking
    return 1 / (1+np.exp((-x+2.5)*20))          # Expanding
stim_plot =[]
for k in range(tmax):
    x = t_trace2[k]  # Shifting the sigmoid function to start at 0
    scale_factor = sigmoid(x)  # Sigmoid expansion factor

    heightloom = int(scale_factor * 100)  # Adjust the initial_height as needed (1 to 150)
    widthloom = int(scale_factor * 100)  # Adjust the initial_width as needed (1 to 150)
    stim_plot.append(heightloom)

    rect_color = (255, 255, 255)  # Color of the rectangle
    rect_top_left = (162 - int(widthloom / 2), 120 - int(heightloom / 2))
    rect_bottom_right = (162 + int(widthloom / 2), 120 + int(heightloom / 2))
    loomfr = cv2.rectangle(loomfr, rect_top_left, rect_bottom_right, rect_color, cv2.FILLED)

    INPUT1.append(loomfr)
    loomfr = dark_base*np.ones((244, 324, 1), np.uint8)
frames = np.expand_dims(np.array(INPUT1), axis=-1)
LPLC2_act2D_loom = [];LC15_act2D_loom = []
for layer_activation in cnn_model.layers:
    if 'LPLC2' == layer_activation.name:
        get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
        layer_activations = get_activations([frames])[0]
        activation = np.squeeze(layer_activations)
        lplc2_gauss = []
        for jj in range(tmax):
            lplc2_trace.append(np.mean(activation[jj,:,:]))
            lplc2_gauss.append(activation[jj, :, :])
        LPLC2_act2D_loom.append(np.mean(np.array(lplc2_gauss), axis=0))
    if 'LC4' == layer_activation.name:
        get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
        layer_activations = get_activations([frames])[0]
        activation = np.squeeze(layer_activations)
        for jj in range(tmax):
            lc4_trace.append(np.mean(activation[jj,:,:]))
    if 'LC15' == layer_activation.name:
        get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
        layer_activations = get_activations([frames])[0]
        activation = np.squeeze(layer_activations)
        lc15_gauss = []
        for jj in range(tmax):
            lc15_trace_loom.append(np.mean(activation[jj,:,:]))
            lc15_gauss.append(activation[jj, :, :])
        LC15_act2D_loom.append(np.mean(np.array(lc15_gauss), axis=0))
    if 'LC11' == layer_activation.name:
        get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
        layer_activations = get_activations([frames])[0]
        activation = np.squeeze(layer_activations)
        for jj in range(tmax):
            lc11_trace_loom.append(np.mean(activation[jj,:,:]))
    if 'classification' == layer_activation.name:
        get_activations = tf.keras.backend.function([cnn_model.input], [layer_activation.output])
        layer_activations = get_activations([frames])[0]
        activation = np.squeeze(layer_activations)
        for jj in range(tmax):
            classif_trace_loom.append(activation[jj, :])
plt.subplot(2, 3, 3)
plt.plot(t_trace2, lplc2_trace, 'green')
plt.plot(t_trace2, lc15_trace_loom, 'b', alpha=0.75)
plt.plot(t_trace2, lc11_trace_loom, 'r', alpha=0.75)
plt.legend(['LPLC2', 'LPLC2 experimental'])
plt.legend(['LPLC2', 'LC15'])
plt.title('Loom')
plt.ylabel('Normalized response (a.u.)')
plt.xlabel('Time (s)')
Rmax_loom = np.max(lplc2_trace/np.max(lplc2_trace))
Rmin_loom = np.max(lc15_trace_loom/np.max(lplc2_trace))

plt.subplot(2, 3, 6)
f_elements = [classif_trace_loom[i][0] for i in range(185)]
plt.plot(t_trace2, f_elements, 'b')
s_elements = [classif_trace_loom[i][1] for i in range(185)]
plt.plot(t_trace2, s_elements, 'r')
t_elements = [classif_trace_loom[i][2] for i in range(185)]
plt.plot(t_trace2, t_elements, 'g')

plt.legend(['Loom', 'Bar', 'Spot'])
########### Selectivity Index ##################
SILC11 = (Rmax_square - Rmin_square) / (Rmax_square + Rmin_square)
SILC15 = (Rmax_rect - Rmin_rect) / (Rmax_rect + Rmin_rect)
SILPLC2 = (Rmax_loom - Rmin_loom) / (Rmax_loom + Rmin_loom)
# Data for the bar plot
################################################
plt.figure()
plt.subplot(1,2,1)
plt.plot(t_trace, T4_b, 'b')
plt.plot(t_trace, T4_b_ND, 'r')
Rmax_T4 = np.max(T4_b / np.max(T4_b))
Rmin_T4 = np.max(T4_b_ND / np.max(T4_b))
SIT4 = (Rmax_T4 - Rmin_T4) / (Rmax_T4 + Rmin_T4)

ymin, ymax = plt.ylim()
plt.legend(['T4 PD', 'T4 ND'])
plt.title('T4 response')
plt.ylabel('Normalized response (a.u.)')
plt.xlabel('Time (s)')
plt.subplot(1,2,2)

plt.plot(t_trace, T5_b, 'b')
plt.plot(t_trace, T5_b_ND, 'r')
Rmax_T5 = np.max(T5_b / np.max(T5_b))
Rmin_T5 = np.max(T5_b_ND / np.max(T5_b))
SIT5 = (Rmax_T5 - Rmin_T5) / (Rmax_T5 + Rmin_T5)

indices = ['T4', 'T5', 'LC11', 'LC15', 'LPLC2']
values = [SIT4, SIT5, SILC11, SILC15, SILPLC2]
plt.legend(['T5 PD', 'T5 ND'])
plt.title('T5 response')
plt.xlabel('Time (s)')

# Create a bar plot
plt.figure()
plt.bar(indices, values)
# Add labels and title
plt.xlabel('Indices')
plt.ylabel('Values')
plt.title('Selectivity Indices')

# Plotting 2D activations averaged over time
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(10, 8))

cmap = 'magma'
axes = [
axs[0,0].imshow(T4_act2D[0]/T4_act2D[0].max(), cmap=cmap, vmin=0, vmax=1),
axs[0,1].imshow(T4ND_act2D[0]/T4_act2D[0].max(), cmap=cmap, vmin=0, vmax=1),
axs[0,2].imshow(T5_act2D[0]/T5_act2D[0].max(), cmap=cmap, vmin=0, vmax=1),
axs[0,3].imshow(T5ND_act2D[0]/T5_act2D[0].max(), cmap=cmap, vmin=0, vmax=1),
axs[1,0].imshow(LC11_act2D[0]/LC15_act2D[0].max(), cmap=cmap, vmin=0, vmax=1),
axs[1,1].imshow(LC15_act2D[0]/LC15_act2D[0].max(), cmap=cmap, vmin=0, vmax=1),
axs[1,2].imshow(LC11_act2D[1]/LC11_act2D[1].max(), cmap=cmap, vmin=0, vmax=1),
axs[1,3].imshow(LC15_act2D[1]/LC11_act2D[1].max(), cmap=cmap, vmin=0, vmax=1),
axs[2,0].imshow(LPLC2_act2D_loom[0]/LPLC2_act2D_loom[0].max(), cmap=cmap, vmin=0, vmax=1),
axs[2,1].imshow(LC15_act2D_loom[0]/LPLC2_act2D_loom[0].max(), cmap=cmap, vmin=0, vmax=1)]

layer_names = ['T4 PD (bright rectangle)', 'T4 ND (bright rectangle)', 'T5 PD (bright rectangle)', 'T5 ND (bright rectangle)', 'LC11 rectangle', 'LC15 rectangle', 'LC11 square', 'LC15 square', 'LPLC2 loom', 'LC15 loom']
for j, ax in enumerate(fig.axes):
    if j < len(layer_names):
        ax.set_title(layer_names[j])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(axes[j], cax=cax)
        #cbar.set_label('Averaged response (a.u.)')
    ax.set_axis_off()

plt.tight_layout()

######################### Showing rest of feature maps #############################
num_activations = len(lamedul_act2D)
# Determine the grid size (rows and columns) for the subplots
grid_size = int(np.ceil(np.sqrt(num_activations)))
# Create a figure to hold the subplots
fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 8))
# Determine the global min and max values for the color scale
global_min = min(act.min() for act in lamedul_act2D)
global_max = max(act.max() for act in lamedul_act2D)
for i, activation in enumerate(lamedul_act2D):
    row = i // grid_size
    col = i % grid_size
    # Display the activation on the corresponding subplot
    ax = axes[row, col]
    if lamedul_names[i] == 'R16':
        cax = ax.matshow(activation[:,:,0]/global_max, cmap='seismic', vmin=-1, vmax=1)
    else:
        cax = ax.matshow(activation/global_max, cmap='seismic', vmin=-1, vmax=1)
    # Add a color bar for each subplot
    fig.colorbar(cax, ax=ax)
    ax.set_title(lamedul_names[i])
    ax.set_xticks([]); ax.set_xticklabels([])
    ax.set_yticks([]); ax.set_yticklabels([])
# Hide any empty subplots
for j in range(i + 1, grid_size * grid_size):
    row = j // grid_size
    col = j % grid_size
    axes[row, col].axis('off')

plt.tight_layout()
####################################################################################
########################### Showing kernels ########################################

total_kernels = sum(weights.shape[2] for weights in conv_weights)
kernel_names = [name.replace("_kernel", "") for name in kernel_names]
# Calculate optimal number of columns and rows for subplots
ncols = int(np.ceil(np.sqrt(total_kernels)))
nrows = int(np.ceil(total_kernels / ncols))

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 8))
axs = axs.flatten()  # Flatten the array of axes for easy indexing
k = 0

# Find the overall minimum and maximum values for color mapping
global_min = min(np.min(weights) for weights in conv_weights)
global_max = max(np.max(weights) for weights in conv_weights)

for i, weights in enumerate(conv_weights):
    for j in range(weights.shape[2]):
        axes = axs[k].matshow(weights[:, :, j, 0]/global_max, cmap='seismic', vmin=-1, vmax=1)
        title = f"{kernel_names[i]}_{j}" if weights.shape[2] > 1 else f"{kernel_names[i]}"
        axs[k].set_title(title)
        axs[k].set_xticks([]); axs[k].set_xticklabels([])
        axs[k].set_yticks([]); axs[k].set_yticklabels([])
        k += 1

# Hide any remaining empty subplots
for ax in axs[k:]:
    ax.set_axis_off()
plt.tight_layout()
################################################################################
plt.show()