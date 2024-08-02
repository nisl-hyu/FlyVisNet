# FlyVisNetH for pruning

# Angel Canelo 2024.08.02

###### import ######################
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Conv2D
##################################
########### CNN Model ############
class filter_cons(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

class FlyVisNetH_pruning:
    def FlyVisNet_model(self, HEIGHT, WIDTH, classes, layer_name):

        pos_constraint = filter_cons(min_value=0.01, max_value=0.5)
        neg_constraint = filter_cons(min_value=-0.5, max_value=-0.01)
        pruned_constraint = filter_cons(min_value=0, max_value=0)

        ## Filter values of for Retina, Lamina, and Medulla layers
        filter_exc = RandomUniform(minval=0.2, maxval=0.3)
        filter_inh = RandomUniform(minval=-0.2, maxval=-0.1)
        filter_inh2 = RandomUniform(minval=-0.2, maxval=-0.1)

        ## Filter values of for Retina, Lamina, and Medulla layers (Option 2)
        # filter_exc = RandomUniform(minval=0.1, maxval=0.2)
        # filter_inh = RandomUniform(minval=-0.25, maxval=-0.2)
        # filter_inh2 = RandomUniform(minval=-0.3, maxval=-0.25)

        filter_T2 = RandomUniform(minval=0.01, maxval=0.15)
        filter_T3 = RandomUniform(minval=-0.15, maxval=-0.01)
        filter_T4 = RandomUniform(minval=0.01, maxval=0.1)
        filter_T5 = RandomUniform(minval=0.01, maxval=0.15)
        filter_LC11 = RandomUniform(minval=0.01, maxval=0.08)
        filter_LC15 = RandomUniform(minval=0.01, maxval=0.04)
        filter_LPLC2 = RandomUniform(minval=0.25, maxval=0.5)
        filter_TmY9 = RandomUniform(minval=0.01, maxval=0.2)
        filter_TmY5 = RandomUniform(minval=0.01, maxval=0.2)

        filter_TmY4 = RandomUniform(minval=0.01, maxval=0.2)

        filter_Li = RandomUniform(minval=-0.15, maxval=-0.01)
        filter_LPi = RandomUniform(minval=-0.1, maxval=-0.05)

        ON_act = 'linear'
        OFF_act = 'linear'
        L_act = 'linear'
        Ts = 'relu'
        T2T3_act = 'linear'
        inter_act = 'linear'
        act = 'relu'

        train = True
        untrain = True

        inputs = layers.Input(shape=[HEIGHT, WIDTH, 1])  # (height, width, channels)

        # RETINA
        R16 = layers.Conv2D(6, 1, data_format="channels_last", kernel_regularizer=l2(1e-3),
                            kernel_initializer=filter_inh,
                            activation=OFF_act, padding='same', name='R16', trainable=train,  kernel_constraint=neg_constraint)(inputs)

        # LAMINA
        L1 = layers.Conv2D(1, 1, data_format="channels_last", kernel_regularizer=l2(1e-3),
                           kernel_initializer=filter_inh,
                           activation=OFF_act, padding='same', name='L1_kernel', trainable=train, kernel_constraint=neg_constraint)(R16)
        L1 = layers.MaxPooling2D(pool_size=(2, 2), name='L1')(L1)
        L1b = layers.MaxPooling2D(pool_size=(2, 2))(L1)
        L2 = layers.Conv2D(1, 1, data_format="channels_last", kernel_regularizer=l2(1e-3),
                           kernel_initializer=filter_exc,
                           activation=L_act, padding='same', name='L2_kernel', trainable=train, kernel_constraint=pos_constraint)(R16)
        L2 = layers.MaxPooling2D(pool_size=(2, 2), name='L2')(L2)
        L2b = layers.MaxPooling2D(pool_size=(2, 2))(L2)
        L3 = layers.Conv2D(1, 1, data_format="channels_last", kernel_regularizer=l2(1e-3),
                           kernel_initializer=filter_exc,
                           activation=L_act, padding='same', name='L3_kernel', trainable=train, kernel_constraint=pos_constraint)(R16)
        L3 = layers.MaxPooling2D(pool_size=(2, 2), name='L3')(L3)
        # MEDULLA
        Mi1 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                            kernel_initializer=filter_exc,
                            activation=ON_act, padding='same', name='Mi1_kernel', trainable=train, kernel_constraint=pos_constraint)(L1)
        Mi1 = layers.MaxPooling2D(pool_size=(2, 2), name='Mi1')(Mi1)

        Tm3 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                            kernel_initializer=filter_exc,
                            activation=ON_act, padding='same', name='Tm3_kernel', trainable=train, kernel_constraint=pos_constraint)(L1)
        Tm3 = layers.MaxPooling2D(pool_size=(2, 2), name='Tm3')(Tm3)

        C3 = layers.concatenate([L1, L3])
        C3 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                           kernel_initializer=filter_inh2,
                           activation=OFF_act, padding='same', name='C3_kernel', trainable=train, kernel_constraint=neg_constraint)(C3)
        C3 = layers.MaxPooling2D(pool_size=(2, 2), name='C3')(C3)
        C3 = layers.Lambda(lambda x: tf.roll(x, shift=1, axis=0))(C3)

        Mi4 = layers.concatenate([L1, L3])
        Mi4 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                            kernel_initializer=filter_inh2,
                            activation=OFF_act, padding='same', name='Mi4_kernel', trainable=train, kernel_constraint=neg_constraint)(Mi4)
        Mi4 = layers.MaxPooling2D(pool_size=(2, 2), name='Mi4')(Mi4)
        Mi4 = layers.Lambda(lambda x: tf.roll(x, shift=1, axis=0))(Mi4)

        Tm1 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                            kernel_initializer=filter_exc,
                            activation=ON_act, padding='same', name='Tm1_kernel', trainable=train, kernel_constraint=pos_constraint)(L2)
        Tm1 = layers.MaxPooling2D(pool_size=(2, 2), name='Tm1')(Tm1)

        Tm2 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                            kernel_initializer=filter_exc,
                            activation=ON_act, padding='same', name='Tm2_kernel', trainable=train, kernel_constraint=pos_constraint)(L2)
        Tm2 = layers.MaxPooling2D(pool_size=(2, 2), name='Tm2')(Tm2)

        Tm4 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                            kernel_initializer=filter_exc,
                            activation=ON_act, padding='same', name='Tm4_kernel', trainable=train, kernel_constraint=pos_constraint)(L2)
        Tm4 = layers.MaxPooling2D(pool_size=(2, 2), name='Tm4')(Tm4)

        Mi9 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                            kernel_initializer=filter_inh2,
                            activation=OFF_act, padding='same', name='Mi9_kernel', trainable=train, kernel_constraint=neg_constraint)(L3)
        Mi9 = layers.MaxPooling2D(pool_size=(2, 2), name='Mi9')(Mi9)
        Mi9 = layers.Lambda(lambda x: tf.roll(x, shift=1, axis=0))(Mi9)

        Tm9 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                            kernel_initializer=filter_exc,
                            activation=ON_act, padding='same', name='Tm9_kernel', trainable=train, kernel_constraint=pos_constraint)(L3)
        Tm9 = layers.MaxPooling2D(pool_size=(2, 2), name='Tm9')(Tm9)
        Tm9 = layers.Lambda(lambda x: tf.roll(x, shift=1, axis=0))(Tm9)

        CT1 = layers.concatenate([L3, L2])
        CT1 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                            kernel_initializer=filter_inh,
                            activation=OFF_act, padding='same', name='CT1_kernel', trainable=train, kernel_constraint=neg_constraint)(CT1)
        CT1 = layers.MaxPooling2D(pool_size=(2, 2), name='CT1')(CT1)
        CT1 = layers.Lambda(lambda x: tf.roll(x, shift=1, axis=0))(CT1)

        TmY9 = layers.concatenate([Tm2, Mi1, C3, Mi4, Tm1, Mi9])
        TmY9 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                             kernel_initializer=filter_TmY9,
                             activation=OFF_act, padding='same', name='TmY9_kernel', trainable=train, kernel_constraint=pos_constraint)(TmY9)
        TmY9 = layers.MaxPooling2D(pool_size=(2, 2), name='TmY9')(TmY9)

        ###
        TmY4 = layers.concatenate([Tm2, Mi1, C3, Mi4, Tm1, Mi9])
        TmY4 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                             kernel_initializer=filter_TmY4,
                             activation=ON_act, padding='same', name='TmY4_kernel', trainable=train, kernel_constraint=pos_constraint)(TmY4)
        TmY4 = layers.MaxPooling2D(pool_size=(2, 2), name='TmY4')(TmY4)
        ###

        TmY5 = layers.concatenate([Mi4, Mi9])
        TmY5 = layers.Conv2D(1, 4, data_format="channels_last", kernel_regularizer=l2(1e-3),
                             kernel_initializer=filter_TmY5,
                             activation=ON_act, padding='same', name='TmY5_kernel', trainable=train, kernel_constraint=pos_constraint)(TmY5)
        TmY5b = layers.MaxPooling2D(pool_size=(2, 2), name='TmY5')(TmY5)

        T2 = layers.concatenate([L1b, Tm2, Mi1, Tm3, C3, Mi4, Tm9, CT1])
        T2 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                           kernel_initializer=filter_T2,
                           activation=T2T3_act, padding='same', name='T2_kernel', trainable=train, kernel_constraint=pos_constraint)(
            T2)
        T2 = layers.MaxPooling2D(pool_size=(2, 2), name='T2')(T2)

        T3 = layers.concatenate([L2b, Tm1, Tm2, Tm3, Mi1, Mi9, C3, Tm9, CT1])
        T3 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                           kernel_initializer=filter_T3,
                           activation=OFF_act, padding='same', name='T3_kernel', trainable=train, kernel_constraint=neg_constraint)(
            T3)
        T3 = layers.MaxPooling2D(pool_size=(2, 2), name='T3')(T3)

        T4 = layers.concatenate([Mi1, Tm3, C3, Mi4, Mi9])
        T4 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                           kernel_initializer=filter_T4,
                           activation=Ts, padding='same', name='T4_kernel', trainable=train, kernel_constraint=pos_constraint)(T4)
        T4 = layers.MaxPooling2D(pool_size=(2, 2), name='T4')(T4)
        # LOBULA
        T5 = layers.concatenate([Tm1, Tm2, Tm4, Tm9, CT1])
        T5 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                           kernel_initializer=filter_T5,
                           activation=Ts, padding='same', name='T5_kernel', trainable=train, kernel_constraint=pos_constraint)(T5)
        T5 = layers.MaxPooling2D(pool_size=(2, 2), name='T5')(T5)

        Li = layers.concatenate([T2, T3])
        Li = layers.Conv2D(1, 5, data_format="channels_last", kernel_regularizer=l2(1e-3),
                                        kernel_initializer=filter_Li,
                                        activation=inter_act, padding='same', name='Li_kernel', trainable=untrain, kernel_constraint=neg_constraint)(Li)

        LPi = layers.concatenate([T4, T5])
        LPi = layers.Conv2D(1, 5, data_format="channels_last", kernel_regularizer=l2(1e-3),
                                         kernel_initializer=filter_LPi,
                                         activation=inter_act, padding='same', name='LPi_kernel', trainable=True, kernel_constraint=neg_constraint)(LPi)

        LC11 = layers.concatenate([T2, T3])
        LC11 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                             kernel_initializer=filter_LC11,
                             activation=act, padding='same', name='LC11_kernel', trainable=untrain, kernel_constraint=pos_constraint)(
            LC11)
        LC11 = layers.MaxPooling2D(pool_size=(2, 2), name='LC11')(LC11)

        LC15 = layers.concatenate([Li, TmY9, TmY4])
        LC15 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                             kernel_initializer=filter_LC15,
                             activation=act, padding='same', name='LC15_kernel', trainable=untrain, kernel_constraint=pos_constraint)(
            LC15)
        LC15 = layers.MaxPooling2D(pool_size=(2, 2), name='LC15')(LC15)

        LPLC2 = layers.concatenate([T4, T5, LPi, TmY5b])
        #LPLC2 = layers.concatenate([T4, T5])
        LPLC2 = layers.Conv2D(1, 3, data_format="channels_last", kernel_regularizer=l2(1e-3),
                              kernel_initializer=filter_LPLC2,
                              activation=act, padding='same', name='LPLC2_kernel', trainable=untrain, kernel_constraint=pos_constraint)(
            LPLC2)
        LPLC2 = layers.MaxPooling2D(pool_size=(2, 2), name='LPLC2')(LPLC2)

        # OPTIC GLOMERULI
        CB = layers.Concatenate()([LC11, LC15, LPLC2])
        CB = layers.Flatten()(CB)
        CB = layers.Dense(128, kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-3),
                           activation='relu')(CB)
        CB = layers.Dense(classes, kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-3),
                           activation='softmax', name='classification')(CB)

        model = Model(inputs=inputs, outputs=CB, name='FlyVisNet')

        # Apply constraint to the specified layer
        self.apply_constraint_to_layer(model, layer_name, pruned_constraint)

        return model

    def apply_constraint_to_layer(self, model, layer_name, pruned_constraint):
        layer = model.get_layer(name=layer_name)
        if isinstance(layer, Conv2D):
            layer.kernel_constraint = pruned_constraint
            layer.bias_constraint = pruned_constraint