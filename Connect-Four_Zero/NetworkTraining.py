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
import math
import random
from Losses import Losses
from C4Game import C4Game
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
        game_start_time = time.time()
        history = None
        losses = Losses()

        clear = True

        for i in range(config.iterations):
            print(f'{BColors.HEADER}Iteration {i}/{config.iterations}{BColors.ENDC}')
            for p in processes:
                p.join()
            print("Self-play games took {} minutes".format((time.time() - game_start_time)/60))

            training_data = replay_buffer.get_batch()
            replay_buffer.clear_buffer() if clear else None
            print(f"Received {len(training_data)} samples of training data and reset buffer.")

            processes = NetworkTraining.collect_game_data(config, replay_buffer)
            game_start_time = time.time()

            train_start_time = time.time()
            new_network, new_history = NetworkTraining.train_network(network.clone_network(config), training_data, config)
            print("Training network took {} minutes".format((time.time() - train_start_time)/60))
            if NetworkTraining.pit_networks(network, new_network, config):
                print(f"{BColors.OKBLUE}Replacing network with new model.{BColors.ENDC}")
                network.cnn.model.set_weights(new_network.cnn.model.get_weights())
                network.cnn.write_weights(config.model_name)
                NetworkTraining.update_losses(new_history, losses, config)
                clear = True
            else:
                clear = False
                NetworkTraining.update_losses(history, losses, config) # unsure if this is good

        for p in processes:
                p.terminate()
        return network

    @staticmethod
    def pit_networks(network: Network, new_network: Network, config: C4Config):
        """Returns True if the new network is better."""
        pitting_start_time = time.time()
        network_wins = 0
        new_network_wins = 0
        tied_games = 0
        for i in range(config.val_games):
            print(f"Playing validation game {i}/{config.val_games}")
            winner = NetworkTraining.play_game_networks([network, new_network], config) # 0 if network, 1 if new_network
            if winner != -100:
                network_wins += 1 - winner
                new_network_wins += winner
            else:
                tied_games += 1
        print(f"Network wins: {network_wins} vs. new network wins: {new_network_wins}. Tied games: {tied_games}")
        print("Pitting networks took {} minutes".format((time.time() - pitting_start_time)/60))
        return new_network_wins * 1.0 > network_wins * config.win_factor

    @staticmethod
    def play_game_networks(networks: list, config: C4Config):
        """Returns 0 if network wins, 1 if new network wins. Returns -100 if it's a tied game."""
        game = C4Game()
        to_play_index = random.randint(0,1)
        first_player = to_play_index
        while not game.terminal() and len(game.history) < config.max_moves:
            _, policy_logits = networks[to_play_index].inference(game.make_image(-1))
            policy = {a: policy_logits[a] for a in game.legal_actions()} # I removed math.exp.
            game.apply(max(policy, key=policy.get)) # gets key for max value
            to_play_index = 1 - to_play_index # switch network
        first_player_win = game.terminal_value(1)
        print(f"first player win value: {first_player_win} with first player with index {first_player}")
        if first_player_win == 0: # tied game
            return -100
        if first_player == 0: # network went first
            if first_player_win == 1:
                return 0 # network wins
            else:
                return 1 # new network wins
        if first_player == 1:
            if first_player_win == 1:
                return 1 # new network wins
            else:
                return 0 # network wins
        return -100 # should be unreachable

    @staticmethod
    def update_losses(history, losses: Losses, config: C4Config):
        if history is None:
            return
        losses.add_loss(history['loss'][config.EPOCHS - 1],\
                            history['value_head_loss'][config.EPOCHS - 1],\
                            history['value_head_loss'][config.EPOCHS - 1])
        

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
        policy_targets = np.array([x[1][1] for x in training_data])
        value_targets = np.array([x[1][0] for x in training_data])
        training_targets = {'value_head': value_targets, 'policy_head': policy_targets}
        print(f"Starting model.fit(...) with batch size: {config.batch_size}.")
        fit = network.cnn.model.fit(x=training_states, y=training_targets, epochs=config.epochs, verbose=1, validation_split=0, batch_size=config.batch_size)
        return network, fit.history

    @staticmethod
    def launch_job(f, *args):
        x = Process(target=f, args=args)
        x.start()
        return x
