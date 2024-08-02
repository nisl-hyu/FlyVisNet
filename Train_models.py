# Train FlyVisNet, Dronet, and MobileNet

# Angel Canelo 2024.08.02

###### import ######################
import tensorflow.keras.callbacks as cb
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
import cv2
import numpy as np
from pymatreader import read_mat
from scipy.io import savemat
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from FlyVisNetH_model import FlyVisNetH
from FlyVisNetL_model import FlyVisNetL
import gc
##################################
class Dronet:
    def res_block(self, x, out_channels):
        identity = x
        x = layers.Conv2D(out_channels, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x_bypass = layers.Conv2D(out_channels, kernel_size=1, strides=2, padding='valid')(identity)
        x += x_bypass
        x = tf.keras.activations.relu(x)
        return x

    def Dronet(self, HEIGHT, WIDTH, classes):
        inputs = layers.Input(shape=[HEIGHT, WIDTH, 1])  # (height, width, channels)
        x = layers.Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        #res_block = ResBlock()
        x = Dronet().res_block(x, 32)
        x = Dronet().res_block(x, 64)
        x = Dronet().res_block(x, 128)

        x = layers.Dropout(0.5)(x)
        x = layers.ReLU()(x)
        x = layers.Flatten()(x)
        # x = layers.Dense(10, activation='relu')(x)
        x = layers.Dense(classes, activation='softmax')(x)

        return Model(inputs=inputs, outputs=x, name='Dronet')
class Dronet2:
    def res_block(self, x, out_channels):
        identity = x
        x = layers.Conv2D(out_channels, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x_bypass = layers.Conv2D(out_channels, kernel_size=1, strides=2, padding='valid')(identity)
        x += x_bypass
        x = tf.keras.activations.relu(x)
        return x

    def Dronet(self, HEIGHT, WIDTH, classes):
        inputs = layers.Input(shape=[HEIGHT, WIDTH, 1])  # (height, width, channels)
        x = layers.Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        #res_block = ResBlock()
        x = Dronet().res_block(x, 32)
        x = Dronet().res_block(x, 64)
        x = Dronet().res_block(x, 128)

        x = layers.Dropout(0.5)(x)
        x = layers.ReLU()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(10, activation='relu')(x)
        x = layers.Dense(classes, activation='softmax')(x)

        return Model(inputs=inputs, outputs=x, name='Dronet')
###############################################
##################################
tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
############### LOAD DATA ###############
standard_dataset = 0   # 0 -> Moving Patterns dataset; 1 -> COIL100
net = 0   # 0 -> FlyVisNet; 1 -> Dronet; 2 -> MobileNetV2
alpha = 0.18    # MobileNet size control hyperparameter
model = 0   # 0 -> 244X324; 1 -> 20X40
model_res = ['244x324', '20x40']
net_name = ['FlyVisNet', 'Dronet', 'MobileNetV2', 'Random_init']
ds_name = ['Pattern', 'COIL100']

gc.collect()
tf.keras.backend.clear_session()
for nnn in range(10):
    if standard_dataset==0:
        chosen = 'Moving_Pattern'
        ######## PREPROCESSING 244X324 ##########
        if model == 0:
            data = read_mat('../data/data_moving_pattern_244x324_train.mat')
            data_test = read_mat('../data/data_moving_pattern_244x324_test.mat')
            HEIGHT = 244
            WIDTH = 324
            n_out = 3
            checkpoint_filepath = '../WEIGHTS/'f"{net_name[net]}_weights_244X324_{chosen}_{nnn}.h5"
            perf_to_file = f"../performance_mat/{net_name[net]}_244X324_{chosen}_perf_loop_{nnn}.mat"
        # TRAIN data
            input_im = np.expand_dims(data['Images'],axis=3)
            norm_images = input_im / 255
            im_label = data['Image_label']
        # TEST data
            input_im_test = np.expand_dims(data_test['Images'],axis=3)
            norm_images_test = input_im_test / 255
            im_label_test = data_test['Image_label']
        ##################################
        ###### PREPROCESSING 20X40 #######
        elif model == 1:
            data = read_mat('../data/data_moving_pattern_20x40_train.mat')
            data_test = read_mat('../data/data_moving_pattern_20x40_test.mat')
            HEIGHT = 20
            WIDTH = 40
            if net == 2:    # Minimum resolution allowed by MobileNet
                HEIGHT = 32
                WIDTH = 32
                alpha = 0.145
            n_out = 3
            checkpoint_filepath = '../WEIGHTS/'f"{net_name[net]}_weights_20X40_{chosen}_{nnn}.h5"
            perf_to_file = f"../performance_mat/{net_name[net]}_20X40_{chosen}_perf_loop_{nnn}.mat"
        # TRAIN data
            input_im = np.expand_dims(data['Images'],axis=3)
            norm_images = input_im / 255
            im_label = data['Image_label']
        # TEST data
            input_im_test = np.expand_dims(data_test['Images'],axis=3)
            norm_images_test = input_im_test / 255
            im_label_test = data_test['Image_label']

    elif standard_dataset==1:
        # Load the COIL-100 dataset
        data_slice = 50000
        x_train = []
        im_label = []
        x_test = []
        im_label_test = []
        ds, info = tfds.load('coil100', split=['train[:75%]', 'train[75%:100%]'], with_info=True)
        for example in ds[0]:
            x_train.append(tfds.as_numpy(example["image"]))
            one_hot_label = tf.keras.utils.to_categorical(tfds.as_numpy(example["object_id"]), num_classes=100)
            im_label.append(one_hot_label)
        for example in ds[1]:
            x_test.append(tfds.as_numpy(example["image"]))
            one_hot_label = tf.keras.utils.to_categorical(tfds.as_numpy(example["object_id"]), num_classes=100)
            im_label_test.append(one_hot_label)
        # Convert images to grayscale
        x_train_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_train]
        x_test_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_test]
        chosen = 'COIL100'
        n_out = 100

        # Resize the grayscale images
        if model==0:
            HEIGHT = 244
            WIDTH = 324
            checkpoint_filepath = '../WEIGHTS/'f"{net_name[net]}_weights_244X324_{chosen}_{nnn}.h5"
            perf_to_file = f"../performance_mat/{net_name[net]}_244X324_{chosen}_perf_loop_{nnn}.mat"
        elif model==1:
            HEIGHT = 20
            WIDTH = 40
            if net == 2:    # Minimum resolution allowed by MobileNet
                HEIGHT = 32
                WIDTH = 32
                alpha = 0.145
            checkpoint_filepath = '../WEIGHTS/'f"{net_name[net]}_weights_20X40_{chosen}_{nnn}.h5"
            perf_to_file = f"../performance_mat/{net_name[net]}_20X40_{chosen}_perf_loop_{nnn}.mat"
        norm_images = np.array([cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_AREA)/255 for img in x_train_gray])
        norm_images_test = np.array([cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_AREA)/255 for img in x_test_gray])
    ###############################################
    if net == 0:
        if model == 0:
            cnn = FlyVisNetH()
            cnn_model = cnn.FlyVisNet_model(HEIGHT, WIDTH, n_out)
        elif model == 1:
            cnn = FlyVisNetL()
            cnn_model = cnn.FlyVisNet_model(HEIGHT, WIDTH, n_out)
    elif net == 1 and standard_dataset == 0:
        cnn_model = Dronet().Dronet(HEIGHT, WIDTH, n_out)
    elif net == 1 and standard_dataset == 1:
        cnn_model = Dronet2().Dronet(HEIGHT, WIDTH, n_out)
    elif net == 2:
        input_shape = (HEIGHT, WIDTH, 1)
        input_tensor = Input(shape=input_shape)
        cnn_model = tf.keras.applications.mobilenet.MobileNet(
            input_tensor=input_tensor,
            alpha=alpha,
            include_top=True,
            weights=None,
            pooling='max',
            classes=n_out)
    #cnn_model.summary()
    print('Trial', nnn+1,  '# Net:', net_name[net], '# Dataset:', chosen, '# Resolution:', model_res[model])
    lr = 1e-3; bz = 25; nb_epochs = 100
    cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
    # define model callbacks
    checkpoint_callback = cb.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy',
                                             mode='max', save_best_only=True)
    cbs = [cb.EarlyStopping(monitor='loss', min_delta=0, patience=100, restore_best_weights=True), checkpoint_callback]
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
