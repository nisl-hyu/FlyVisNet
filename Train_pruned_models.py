# Train Pruning by layer FlyVisNet

# Angel Canelo 2024.08.02

###### import ######################
import tensorflow.keras.callbacks as cb
import numpy as np
from pymatreader import read_mat
from scipy.io import savemat
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from FlyVisNetH_pruning_model import FlyVisNetH_pruning
import gc
##################################
###############################################
##################################
tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
############### LOAD DATA ###############
layer_names = ['L1_kernel', 'L2_kernel', 'L3_kernel', 'Mi1_kernel', 'Tm3_kernel', 'C3_kernel', 'Mi4_kernel', 'Tm1_kernel',
               'Tm2_kernel', 'Tm4_kernel', 'Mi9_kernel', 'Tm9_kernel', 'CT1_kernel', 'TmY9_kernel', 'TmY4_kernel', 'TmY5_kernel',
               'T2_kernel', 'T3_kernel', 'T4_kernel', 'T5_kernel', 'Li_kernel', 'LPi_kernel', 'LC11_kernel', 'LC15_kernel', 'LPLC2_kernel']
run = 10
for kkk in range(10):
    for nnn in range(len(layer_names)):
        chosen = 'Moving_Pattern'
        ######## PREPROCESSING 244X324 ##########
        data = read_mat('../data/data_moving_pattern_244x324_train.mat')
        data_test = read_mat('../data/data_moving_pattern_244x324_test.mat')
        HEIGHT = 244
        WIDTH = 324
        n_out = 3
        perf_to_file = f"../performance_mat/PRUNING/FlyVisNet_244X324_{chosen}_perf_loop_{layer_names[nnn]}_{kkk}.mat"
    # TRAIN data
        input_im = np.expand_dims(data['Images'],axis=3)
        norm_images = input_im / 255
        im_label = data['Image_label']
    # TEST data
        input_im_test = np.expand_dims(data_test['Images'],axis=3)
        norm_images_test = input_im_test / 255
        im_label_test = data_test['Image_label']
        ###############################################
        cnn = FlyVisNetH_pruning()
        cnn_model = cnn.FlyVisNet_model(HEIGHT, WIDTH, n_out, layer_names[nnn])
        print('#', kkk+1, 'run of', run, '#', nnn+1, 'of', len(layer_names), '-- Pruned layer:', layer_names[nnn])

        lr = 1e-3; bz = 25; nb_epochs = 100
        cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
        # define model callbacks
        cbs = [cb.EarlyStopping(monitor='loss', min_delta=0, patience=100, restore_best_weights=True)]
        # train
        history = cnn_model.fit(norm_images, np.array(im_label), batch_size=bz, epochs=nb_epochs,
                              callbacks=cbs, validation_data=(norm_images_test, np.array(im_label_test)))
        gc.collect()
        tf.keras.backend.clear_session()
        #########################################
        ######## Saving results ###############
        to_mat = {"hist_acc": history.history['accuracy'], "hist_testacc": history.history['val_accuracy'],
                  "topmax": np.max(history.history['accuracy']), "topmax_test": np.max(history.history['val_accuracy']),
                  "Epoch #": np.argmax(history.history['val_accuracy'])}
        savemat(perf_to_file,  to_mat)

        print('Top acc', np.max(history.history['val_accuracy']), 'Epoch #', np.argmax(history.history['val_accuracy']))
