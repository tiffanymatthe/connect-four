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


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class NetworkTraining(object):

    @staticmethod
    def collect_game_data(config, network, replay_buffer):
        processes = []
        for _ in range(config.num_actors):
            p = NetworkTraining.launch_job(
                SelfPlay.run_selfplay, config, network, replay_buffer)
            processes.append(p)

        for p in processes:
            p.join()

    @staticmethod
    def alphazero(config: C4Config):
        tf.keras.backend.clear_session()
        replay_buffer = NetworkTraining.get_buffer_from_base_manager(
            config)

        network = Network()

        for _ in config.training_steps:
            NetworkTraining.collect_game_data(config, network, replay_buffer)
            training_data = replay_buffer.get_batch()
            replay_buffer.clear_buffer()
            network = NetworkTraining.train_network(network, training_data, config)
            # maybe add comparison of which one is better (check losses or history or accuracy?)
        return network

    @staticmethod
    def get_buffer_from_base_manager(config):
        BaseManager.register('ReplayBuffer', ReplayBuffer)
        manager = BaseManager()
        manager.start()
        replay_buffer = manager.ReplayBuffer(config)
        return replay_buffer

    # ----------------TRAINING------------------------------------------
    @staticmethod
    def get_learning_rate_fn(config: C4Config):
        boundaries = list(config.learning_rate_schedule.keys())
        boundaries.pop(0)
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, config.learning_rate_schedule.values())

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

        network.fit(...) # TODO

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
