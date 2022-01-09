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
    def collect_game_data(config, replay_buffer, random=False):
        processes = []
        for _ in range(config.num_actors):
            p = NetworkTraining.launch_job(
                SelfPlay.run_selfplay, config, replay_buffer, random)
            processes.append(p)

        return processes

    @staticmethod
    def alphazero(config: C4Config, load: bool):
        start_time = time.time()
        tf.keras.backend.clear_session()
        replay_buffer = NetworkTraining.get_buffer_from_base_manager(config)

        network = Network(config)
        network.cnn.set_learning_rate(config.learning_rate_schedule[0])

        if load:
            network.cnn.read_weights(config.model_name)
        else:
            network.cnn.write_weights(config.model_name)

        processes = NetworkTraining.collect_game_data(config, replay_buffer, True)
        game_start_time = time.time()
        losses = Losses()
        if load:
            try:
                losses.get_losses(config.model_name)
            except FileNotFoundError:
                print("Losses file not found.")

        for i in range(config.iterations):
            print(f'{BColors.HEADER}Iteration {i}/{config.iterations}{BColors.ENDC}')
            for p in processes:
                p.join()
            print("Self-play games took {} minutes".format((time.time() - game_start_time)/60))

            training_data = replay_buffer.sample_batch()
            print(f"Received {len(training_data)} samples of training data.")

            rand = i < config.random_iterations
            processes = NetworkTraining.collect_game_data(config, replay_buffer, rand)
            game_start_time = time.time()

            if i in config.learning_rate_schedule.keys():
                lr = config.learning_rate_schedule[i]
                print(f"Updated learning rate to {lr}")
                network.cnn.set_learning_rate(lr)

            train_start_time = time.time()
            print(network.get_weights())
            _, new_history = NetworkTraining.train_network(network, training_data, config)
            print(network.get_weights())
            print("Training network took {} minutes".format((time.time() - train_start_time)/60))
            NetworkTraining.update_losses(new_history, losses, config)
            network.cnn.write_weights(config.model_name)
            if i % config.checkpoint_interval == 0:
                print(f'{BColors.UNDERLINE}Saving weights at checkpoint interval.{BColors.ENDC}')
                network.cnn.write_weights(f"{config.model_name}_iteration_{i}")
            losses.save(config.model_name)

        for p in processes:
                p.terminate()

        print("Total time is {} minutes".format((time.time() - start_time)/60))
        return network

    @staticmethod
    def pit_networks(network: Network, new_network: Network, config: C4Config):
        """Returns True if the new network is better."""
        pitting_start_time = time.time()
        network_wins = 0
        new_network_wins = 0
        tied_games = 0
        for i in range(config.val_games):
            # print(f"Playing validation game {i}/{config.val_games}")
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
    def softmax_policy_sample(policy: dict):
        """policy: dictionary of actions (keys) and associated logit"""
        policy_sum = sum(policy.values())
        probs = [val / policy_sum for val in policy.values()]
        idx = np.where(np.random.multinomial(1, probs) == 1)[0][0]
        return list(policy.keys())[idx]

    @staticmethod
    def play_game_networks(networks: list, config: C4Config):
        """Returns 0 if network wins, 1 if new network wins. Returns -100 if it's a tied game."""
        game = C4Game()
        to_play_index = random.randint(0,1)
        first_player = to_play_index
        rand = random.randint(1,config.val_games)
        display = rand == 1 or rand == 2
        while not game.terminal() and len(game.history) < config.max_moves:
            _, policy_logits = networks[to_play_index].inference(game.make_image(-1))
            policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
            if len(game.history) < config.num_sampling_moves:
                game.apply(NetworkTraining.softmax_policy_sample(policy))
            else:
                game.apply(max(policy, key=policy.get))
            to_play_index = 1 - to_play_index # switch network
        first_player_win = game.terminal_value(1)

        if display:
            print(f"First player index:{first_player}")
            game.see_board()

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
        losses.add_loss(history['loss'][config.epochs - 1],\
                            history['value_head_loss'][config.epochs - 1],\
                            history['value_head_loss'][config.epochs - 1])
        

    @staticmethod
    def get_buffer_from_base_manager(config: C4Config):
        BaseManager.register('ReplayBuffer', ReplayBuffer)
        manager = BaseManager()
        manager.start()
        replay_buffer = manager.ReplayBuffer(config)
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

# if __name__ == "__main__":
#     config = C4Config()
#     network = Network(config)
#     network1 = Network(config)

#     NetworkTraining.play_game_networks([network, network1], config)