# Global pruning FlyVisNetH

# Angel Canelo 2024.08.02

###### import ######################
import numpy as np
from pymatreader import read_mat
from scipy.io import savemat
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_model_optimization as tfmot
import tempfile
from FlyVisNetH_model import FlyVisNetH
import gc
##################################
###############################################
##################################
tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.keras.backend.clear_session()
############### LOAD DATA ###############
######## PREPROCESSING 244X324 ##########
data = read_mat('../data/data_moving_pattern_244x324_train.mat')
data_test = read_mat('../data/data_moving_pattern_244x324_test.mat')
HEIGHT = 244
WIDTH = 324
n_out = 3
# TRAIN data
input_im = np.expand_dims(data['Images'],axis=3)
norm_images = input_im / 255
im_label = data['Image_label']
# TEST data
input_im_test = np.expand_dims(data_test['Images'],axis=3)
norm_images_test = input_im_test / 255
im_label_test = data_test['Image_label']
##################################
###############################################
### Pruning model #########################################################
prun_glob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
iterat = 10
for kkk in range(iterat):
    for nnn in range(len(prun_glob)):
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        batch_size = 25
        #### Global pruning ##########
        epochs = 1
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(prun_glob[nnn], 0)}
        cnn = FlyVisNetH()
        cnn_model = cnn.FlyVisNet_model(HEIGHT, WIDTH, n_out)
        # Load Pre-trained model
        cnn_model.load_weights(f"../WEIGHTS/FlyVisNet_weights_244X324_Moving_Pattern_0.h5")
        model_for_pruning = prune_low_magnitude(cnn_model, **pruning_params)
        logdir = tempfile.mkdtemp()
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]
        perf_to_file = f"../performance_mat/FlyVisNet_pruned_global_perf_{prun_glob[nnn]}_{kkk}.mat"
        #################################
        print('#', kkk + 1, 'of', iterat, 'Global pruning with sparsity:', prun_glob[nnn])
        lr = 1e-3
        model_for_pruning.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
        # model_for_pruning.summary()

        history_pruned = model_for_pruning.fit(norm_images, np.array(im_label), batch_size=batch_size, epochs=epochs,
                          callbacks=callbacks, validation_data=(norm_images_test, np.array(im_label_test)))
        gc.collect()
        tf.keras.backend.clear_session()
        #####################################################################################
        ######## Plotting results ###############
        to_mat = {"hist_acc": history_pruned.history['accuracy'], "hist_testacc": history_pruned.history['val_accuracy'],
                  "topmax": np.max(history_pruned.history['accuracy']), "topmax_test": np.max(history_pruned.history['val_accuracy']),
                  "Epoch #": np.argmax(history_pruned.history['val_accuracy'])}
        savemat(perf_to_file,  to_mat)

        print('Top acc', np.max(history_pruned.history['val_accuracy']), 'Epoch #', np.argmax(history_pruned.history['val_accuracy']))