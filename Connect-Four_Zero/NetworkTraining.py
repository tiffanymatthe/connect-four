#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from multiprocessing.managers import BaseManager
from multiprocessing import Process
import tensorflow as tf
from SelfPlay import SelfPlay
from BColors import BColors
from Network import Network
from ReplayBuffer import ReplayBuffer
from C4Config import C4Config
import numpy as np
from Losses import Losses
import time
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class NetworkTraining(object):

    @staticmethod
    def collect_game_data(config, replay_buffer):
        processes = []
        for _ in range(config.num_actors):
            p = NetworkTraining.launch_job(
                SelfPlay.run_selfplay, config, replay_buffer)
            processes.append(p)

        return processes

    @staticmethod
    def alphazero(config: C4Config):
        tf.keras.backend.clear_session()
        replay_buffer = NetworkTraining.get_buffer_from_base_manager()
        network = Network(config)

        network.cnn.write_weights(config.model_name)

        processes = NetworkTraining.collect_game_data(config, replay_buffer)
        history = None
        losses = Losses()

        for i in range(config.iterations):
            print(f'{BColors.HEADER}Iteration {i}/{config.iterations}{BColors.ENDC}')
            for p in processes:
                p.join()
            training_data = replay_buffer.get_batch()
            replay_buffer.clear_buffer()
            print("Received training data and reset buffer.")
            processes = NetworkTraining.collect_game_data(config, replay_buffer)
            new_network, new_history = NetworkTraining.train_network(network.clone_network(config), training_data, config)
            if NetworkTraining.pit_networks(history, new_history, losses, config):
                print(f"{BColors.OKBLUE}Replacing network with new model.{BColors.ENDC}")
                network.cnn.model.set_weights(new_network.cnn.model.get_weights())
                network.cnn.write_weights(config.model_name)
        return network

    @staticmethod
    def pit_networks(history, new_history, losses: Losses, config: C4Config):
        """Returns True if the new network is better. Also updates overall losses."""
        if history is None:
            return True
        overall_loss = history['loss'][config.EPOCHS - 1]
        new_overall_loss = new_history['loss'][config.EPOCHS - 1]
        update_network = False
        if new_overall_loss <= overall_loss:
            update_network = True
            losses.add_loss(new_history['loss'][config.EPOCHS - 1],\
                            new_history['value_head_loss'][config.EPOCHS - 1],\
                            new_history['value_head_loss'][config.EPOCHS - 1])
        else:
            losses.add_loss(history['loss'][config.EPOCHS - 1],\
                            history['value_head_loss'][config.EPOCHS - 1],\
                            history['value_head_loss'][config.EPOCHS - 1])
        print(losses.losses)
        return update_network
        

    @staticmethod
    def get_buffer_from_base_manager():
        BaseManager.register('ReplayBuffer', ReplayBuffer)
        manager = BaseManager()
        manager.start()
        replay_buffer = manager.ReplayBuffer()
        return replay_buffer

    @staticmethod
    def train_network(network: Network, training_data: list, config: C4Config):
        """training_data is a list with tuples (image, (policy, value))"""
        print("Collecting inputs to training.")
        training_states = np.array([x[0] for x in training_data])
        policy_targets = np.array([x[1][0] for x in training_data])
        print(policy_targets)
        value_targets = np.array([x[1][1] for x in training_data])
        training_targets = {'value_head': value_targets, 'policy_head': policy_targets}
        print("Starting model.fit(...).")
        fit = network.cnn.model.fit(x=training_states, y=training_targets, epochs=config.epochs, verbose=1, validation_split=0, batch_size=config.batch_size)
        return network, fit.history

    @staticmethod
    def launch_job(f, *args):
        x = Process(target=f, args=args)
        x.start()
        return x
