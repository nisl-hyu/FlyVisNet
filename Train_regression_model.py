# Train FlyVisNetH for regression and convert to TFLite model

# Angel Canelo 2024.08.02

###### import ######################
import tensorflow.keras.callbacks as cb
import numpy as np
from pymatreader import read_mat
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from FlyVisNetH_regression_model import FlyVisNetH_regression
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
gc.collect()
tf.keras.backend.clear_session()
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
X = data['X']
# TEST data
input_im_test = np.expand_dims(data_test['Images'],axis=3)
norm_images_test = input_im_test / 255
im_label_test = data_test['Image_label']
X_test = data_test['X']
##################################
cnn = FlyVisNetH_regression()
cnn_model = cnn.FlyVisNet_model(HEIGHT, WIDTH, n_out)
#cnn_model.summary()
lr = 1e-3; bz = 25; nb_epochs = 100
cnn_model.compile(loss={'classification': 'categorical_crossentropy', 'X': 'mse'},
                loss_weights={'classification': 0.5, 'X': 1.0}, optimizer=Adam(lr),
                metrics={'classification': 'accuracy', 'X': 'mse'})
# define model callbacks
checkpoint_filepath = '../WEIGHTS/FlyVisNet_regression_weights.h5'
checkpoint_callback = cb.ModelCheckpoint(filepath=checkpoint_filepath, monitor='loss',
                                    mode='min', save_best_only=True)
cbs = [cb.EarlyStopping(monitor='X_loss', min_delta=0, patience=100, restore_best_weights=True), checkpoint_callback]
# train
history = cnn_model.fit(norm_images, {'classification': np.array(im_label), 'X': X}, batch_size=bz, epochs=nb_epochs,
                      callbacks=cbs, validation_data=(norm_images_test, {'classification': np.array(im_label_test), 'X': X_test}))
#########################################
######## Results ###############
print('Top acc', np.max(history.history['val_accuracy']), 'Epoch #', np.argmax(history.history['val_accuracy']))
###### Convert to TFLite model for GAP8######
cnn_model.load_weights("../WEIGHTS/FlyVisNet_regression_weights.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
tflite_model = converter.convert()
# save the model
filetf = "../WEIGHTS/classification_q.tflite"
open(filetf, "wb").write(tflite_model)
