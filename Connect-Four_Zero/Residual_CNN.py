#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.disable(logging.WARNING)
from C4Config import C4Config
from Softmax_Loss import softmax_cross_entropy_with_logits 
from tensorflow.keras.optimizers import SGD
from keras import regularizers
from BColors import BColors
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
Modified by https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188
"""

class Gen_Model():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def write(self, version):
        self.model.save('models/version' + "{0:0>4}".format(version) + '.h5')
        print(f'{BColors.OKGREEN}Saved model.{BColors.ENDC}')

    def write_weights(self, version):
        self.model.save_weights('weights/version' + "{0:0>4}".format(version) + '.h5')
        print(f'{BColors.OKGREEN}Saved weights.{BColors.ENDC}')

    def read_weights(self, version):
        self.model.load_weights('weights/version' + "{0:0>4}".format(version) + '.h5')
        print(f'{BColors.OKGREEN}Retrieved weights.{BColors.ENDC}')

    def read(self, version):
        return load_model("models/version" + "{0:0>4}".format(version) + '.h5', custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})


class Residual_CNN(Gen_Model):
    def __init__(self, config: C4Config, model_name=None, model=None):
        Gen_Model.__init__(self, config.input_shape, config.output_policy_shape)
        self.hidden_layers = config.hidden_layers
        self.reg_const = config.weight_decay
        self.num_layers = len(self.hidden_layers)
        self.learning_rate = 0.01
        self.model = None
        if model:
            self.model = model
        elif model_name:
            self.model = self._initialize_model()
            self.read_weights(model_name)
        else:
            self.model = self._build_model(config)

    def set_learning_rate(self, lr: float):
        from keras import backend as K
        self.learning_rate = lr
        K.set_value(self.model.optimizer.learning_rate, lr)

    def residual_layer(self, input_block, filters, kernel_size):
        x = self.conv_layer(input_block, filters, kernel_size)

        x = Conv2D(
            filters=filters, kernel_size=kernel_size, padding='same', 
            use_bias=False, activation='linear', 
            kernel_regularizer = regularizers.l2(self.reg_const))(x)

        x = BatchNormalization(axis=-1)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)

        return (x)

    def conv_layer(self, x, filters, kernel_size):
        x = Conv2D(
            filters=filters, kernel_size=kernel_size, padding='same', 
            use_bias=False, activation='linear')(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)

        return (x)

    def value_head(self, x):

        x = Conv2D(
            filters=1, kernel_size=(1, 1), padding='same', use_bias=False, 
            activation='linear', 
            kernel_regularizer = regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(
            20, use_bias=False, activation='linear', 
            kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = LeakyReLU()(x)
        x = Dense(
            1, use_bias=False, activation='tanh', 
            kernel_regularizer=regularizers.l2(self.reg_const), 
            name='value_head')(x)

        return (x)

    def policy_head(self, x):

        x = Conv2D(
            filters=2, kernel_size=(1, 1), padding='same', use_bias=False, 
            activation='linear', 
            kernel_regularizer = regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(
            self.output_dim, use_bias=False, activation='linear', 
            kernel_regularizer=regularizers.l2(self.reg_const), 
            name='policy_head')(x)

        return (x)

    def _initialize_model(self):
        """Not compiled. Only for inference (MCTS stuff)"""
        main_input = Input(shape=self.input_dim, name='main_input')

        x = self.conv_layer(
            main_input, self.hidden_layers[0]['filters'], 
            self.hidden_layers[0]['kernel_size'])

        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self.residual_layer(x, h['filters'], h['kernel_size'])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        return Model(inputs=[main_input], outputs=[vh, ph])

    def _build_model(self, config:C4Config):

        model = self._initialize_model()
        losses = {
            "value_head": "mse", 
            "policy_head": softmax_cross_entropy_with_logits
        }
        opt = SGD(learning_rate = self.learning_rate, 
                momentum = config.momentum)
        model.compile(loss = losses,
                    optimizer = opt,
                    loss_weights = {'value_head': 0.5, 'policy_head': 0.5})

        return model
