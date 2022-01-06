#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from typing import List
import math
import multiprocessing
from C4Node import C4Node
from C4Game import C4Game
from Network import Network
from ReplayBuffer import ReplayBuffer
from C4Config import C4Config


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class SelfPlay():

    # Each self-play job is independent of all others; it takes the latest network
    # snapshot, produces a game and makes it available to the training job by
    # writing it to a shared replay buffer.

    @staticmethod
    def run_selfplay(config: C4Config, replay_buffer: ReplayBuffer):
        tf.keras.backend.clear_session()
        network = Network(config, model_name=config.model_name) # loads model from files
        id = multiprocessing.current_process()._identity[0]
        print("Starting self-play for process {}".format(id))
        while replay_buffer.get_buffer_size() < config.num_games:
            game = SelfPlay.play_game(config, network)
            replay_buffer.save_game(game)
            print("Game {}/{} finished by process {}".format(replay_buffer.get_buffer_size(), config.num_games, id))

    @staticmethod
    def run_selfplay_main(config: C4Config, replay_buffer: ReplayBuffer):
        print("before initializing model")
        network = Network(config, model_name=config.model_name) # loads model from files
        print("Starting self-play for process")
        while replay_buffer.get_buffer_size() < config.num_games:
            game = SelfPlay.play_game(config, network)
            replay_buffer.save_game(game)
            print("Game {}/{} finished.".format(replay_buffer.get_buffer_size(), config.num_games))

    # Each game is produced by starting at the initial board position, then
    # repeatedly executing a Monte Carlo Tree Search to generate moves until the end
    # of the game is reached.

    @staticmethod
    def play_game(config: C4Config, network: Network):
        game = C4Game()
        while not game.terminal() and len(game.history) < config.max_moves:
            print(len(game.history))
            action, root = SelfPlay.run_mcts(config, game, network)
            game.apply(action)
            game.store_search_statistics(root)
        return game

    # Core Monte Carlo Tree Search algorithm.
    # To decide on an action, we run N simulations, always starting at the root of
    # the search tree and traversing the tree according to the UCB formula until we
    # reach a leaf node.

    @staticmethod
    def run_mcts(config: C4Config, game: C4Game, network: Network):
        root = C4Node(0)
        SelfPlay.evaluate(root, game, network)
        SelfPlay.add_exploration_noise(config, root)

        for i in range(config.num_simulations):
            print(i)
            node = root
            scratch_game = game.clone()
            search_path = [node]

            while node.expanded():
                action, node = SelfPlay.select_child(config, node)
                scratch_game.apply(action)
                search_path.append(node)
                value = SelfPlay.evaluate(node, scratch_game, network)
                SelfPlay.backpropagate(
                    search_path, value, scratch_game.to_play())
        return SelfPlay.select_action(config, game, root), root

    @staticmethod
    def select_action(config: C4Config, game: C4Game, root: C4Node):
        visit_counts = [(child.visit_count, action)
                        for action, child in root.children.items()]

        if len(game.history) < config.num_sampling_moves:
            _, action = SelfPlay.softmax_sample(visit_counts)
        else:
            _, action = max(visit_counts, key=lambda x:x[0])
        return action

    # Select the child with the highest UCB score.

    @staticmethod
    def select_child(config: C4Config, node: C4Node):
        _, action, child = max((SelfPlay.ucb_score(config, node, child), action, child)
                               for action, child in node.children.items())
        return action, child

    # The score for a node is based on its value, plus an exploration bonus based on
    # the prior.

    @staticmethod
    def ucb_score(config: C4Config, parent: C4Node, child: C4Node):
        pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                        config.pb_c_base) + config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = child.value()
        return prior_score + value_score

    # We use the neural network to obtain a value and policy prediction.

    @staticmethod
    def evaluate(node: C4Node, game: C4Game, network: Network):
        value, policy_logits = network.inference(game.make_image(-1))
        print(value)
        print(policy_logits)
        # Expand the node.
        node.to_play = game.to_play()
        policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = C4Node(p / policy_sum)
        return value

    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.

    @staticmethod
    def backpropagate(search_path: List[C4Node], value: float, to_play):
        for node in search_path:
            node.value_sum += value if node.to_play == to_play else (1 - value)
            node.visit_count += 1

    # At the start of each search, we add dirichlet noise to the prior of the root
    # to encourage the search to explore new actions.

    @staticmethod
    def add_exploration_noise(config: C4Config, node: C4Node):
        actions = node.children.keys()
        noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
        frac = config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * \
                (1 - frac) + n * frac

    @staticmethod
    def softmax_sample(d):
        """
        d: list of tuples (visit_count, associated action)
        Returns softmax visit count and action.
        """
        visit_counts = [t[0] for t in d]
        visit_count_probabilities = visit_counts / np.sum(visit_counts)
        idx = np.random.multinomial(1, visit_count_probabilities)
        return d[np.where(idx == 1)[0][0]]