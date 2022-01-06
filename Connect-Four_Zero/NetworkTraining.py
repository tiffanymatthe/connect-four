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

        network.cnn.write(config.model_name)

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
            new_network, new_history = NetworkTraining.train_network(network.clone_network(), training_data, config)
            if NetworkTraining.pit_networks(history, new_history, losses, config):
                print(f"{BColors.OKBLUE}Replacing network with new model.{BColors.ENDC}")
                network.cnn.model.set_weights(new_network.cnn.model.get_weights())
                network.cnn.write(config.model_name)
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

    # ----------------TRAINING------------------------------------------
    # @staticmethod
    # def get_learning_rate_fn(config: C4Config):
    #     boundaries = list(config.learning_rate_schedule.keys())
    #     boundaries.pop(0)
    #     return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    #         boundaries, config.learning_rate_schedule.values())

    # @staticmethod
    # def save_at_checkpoint(replay_buffer: ReplayBuffer, storage: SharedStorage,
    #                        losses: Losses, config: C4Config, step: int, network: Network):
    #     print("{}At checkpoint {}/{}{}".format(BColors.OKBLUE,
    #                                            step, config.training_steps, BColors.ENDC))
    #     print("Replay buffer size: {}".format(
    #         replay_buffer.get_buffer_size()))
    #     storage.save_network(step, network, config)
    #     losses.save(f"losses_{config.model_name}")
    #     losses.print_losses()
    #     network.cnn.write(config.model_name)
    #     print("Saved and downloaded neural network model and losses.")

    @staticmethod
    def train_network(network: Network, training_data: list, config: C4Config):
        """training_data is a list with tuples (image, (policy, value))"""
        training_states = np.array([x[0] for x in training_data])
        policy_targets = np.array([x[1][0] for x in training_data])
        value_targets = np.array([x[1][1] for x in training_data])
        training_targets = {'value_head': value_targets, 'policy_head': policy_targets}
        fit = network.cnn.model.fit(x=training_states, y=training_targets, epochs=config.epochs, verbose=1, validation_split=0, batch_size=config.batch_size)
        return network, fit.history

    # @staticmethod
    # def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: Network, batch,
    #                    weight_decay: float, losses: Losses):
    #     def loss_fcn():
    #         loss = 0
    #         mse = tf.keras.losses.MeanSquaredError(reduction="auto")
    #         for image, (target_value, target_policy) in batch:
    #             value, policy_logits = network.inference(image)
    #             target_value = [target_value]
    #             loss += (
    #                 mse(value, target_value).numpy() +
    #                 tf.nn.softmax_cross_entropy_with_logits(
    #                     logits=policy_logits, labels=target_policy))

    #         for weights in network.get_weights():
    #             loss += weight_decay * tf.nn.l2_loss(weights)

    #         losses.add_loss(loss)

    #         return loss

    #     optimizer.minimize(loss_fcn, var_list=network.get_weights())

    @staticmethod
    def launch_job(f, *args):
        x = Process(target=f, args=args)
        x.start()
        return x
