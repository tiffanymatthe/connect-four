#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.disable(logging.WARNING)
from C4Config import C4Config

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

    def read(self, version):
        return load_model("models/version" + "{0:0>4}".format(version) + '.h5')


class Residual_CNN(Gen_Model):
    def __init__(self, config: C4Config, model=None):
        Gen_Model.__init__(self, config.input_shape, config.output_policy_shape)
        self.hidden_layers = config.hidden_layers
        self.num_layers = len(self.hidden_layers)
        self.model = model if model else self._build_model()

    def residual_layer(self, input_block, filters, kernel_size):

        x = self.conv_layer(input_block, filters, kernel_size)

        x = Conv2D(
            filters=filters, kernel_size=kernel_size, padding='same', use_bias=False, activation='linear'
        )(x)

        x = BatchNormalization(axis=-1)(x)

        x = add([input_block, x])

        x = LeakyReLU()(x)

        return (x)

    def conv_layer(self, x, filters, kernel_size):

        x = Conv2D(
            filters=filters, kernel_size=kernel_size, padding='same', use_bias=False, activation='linear'
        )(x)

        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)

        return (x)

    def value_head(self, x):

        x = Conv2D(
            filters=1, kernel_size=(1, 1), padding='same', use_bias=False, activation='linear'
        )(x)

        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(
            20, use_bias=False, activation='linear'
        )(x)

        x = LeakyReLU()(x)

        x = Dense(
            1, use_bias=False, activation='tanh', name='value_head'
        )(x)

        return (x)

    def policy_head(self, x):

        x = Conv2D(
            filters=2, kernel_size=(1, 1), padding='same', use_bias=False, activation='linear'
        )(x)

        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(
            self.output_dim, use_bias=False, activation='linear', name='policy_head'
        )(x)

        return (x)

    def get_learning_rate_fn(self, config: C4Config):
        boundaries = list(config.learning_rate_schedule.keys())
        boundaries.pop(0)
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, config.learning_rate_schedule.values())

    def _build_model(self):

        main_input = Input(shape=self.input_dim, name='main_input')

        x = self.conv_layer(
            main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self.residual_layer(x, h['filters'], h['kernel_size'])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        model = Model(inputs=[main_input], outputs=[vh, ph])

        return model
    
    def compile_model(self):
        
