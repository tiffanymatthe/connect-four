#!/usr/bin/env python
from C4Config import C4Config
from SharedStorage import SharedStorage
from ReplayBuffer import ReplayBuffer
from Network import Network
from C4Game import C4Game
from C4Node import C4Node
from BColors import BColors

import math
import random
import time
import multiprocessing
from typing import List
import numpy as np
import tensorflow as tf
from multiprocessing import Process
from multiprocessing.managers import BaseManager
import random

class NetworkTraining(object):

    @staticmethod
    def alphazero(config: C4Config):
        replay_buffer, storage = NetworkTraining.get_buffer_storage_from_base_manager(
            config)

        processes = []

        for _ in range(config.num_actors):
            p = NetworkTraining.launch_job(
                NetworkTraining.run_selfplay, config, storage, replay_buffer)
            processes.append(p)

        NetworkTraining.train_network(config, storage, replay_buffer)

        for p in processes:
            p.terminate()

        return storage.latest_network()

    @staticmethod
    def get_buffer_storage_from_base_manager(config):
        BaseManager.register('ReplayBuffer', ReplayBuffer)
        BaseManager.register('SharedStorage', SharedStorage)
        manager = BaseManager()
        manager.start()
        replay_buffer = manager.ReplayBuffer(config)
        storage = manager.SharedStorage()
        return replay_buffer, storage

    # ----------------SELF PLAY----------------------------------------

    # Each self-play job is independent of all others; it takes the latest network
    # snapshot, produces a game and makes it available to the training job by
    # writing it to a shared replay buffer.

    @staticmethod
    def run_selfplay(config: C4Config, storage: SharedStorage,
                     replay_buffer: ReplayBuffer):
        i = 0
        id = multiprocessing.current_process()._identity[0]
        print("Starting self-play for process {}".format(id))
        while True:
            for _ in range(3):
                try:
                    network = storage.latest_network()
                    break
                except KeyError:
                    print(f"{BColors.WARNING}Key Error when trying to retrieve latest network. Trying again.{BColors.ENDC}")
            game = NetworkTraining.play_game(config, network)
            replay_buffer.save_game(game)
            print("Finished game {} for process {}".format(i, id))
            i += 1

    # Each game is produced by starting at the initial board position, then
    # repeatedly executing a Monte Carlo Tree Search to generate moves until the end
    # of the game is reached.

    @staticmethod
    def play_game(config: C4Config, network: Network):
        game = C4Game()
        while not game.terminal() and len(game.history) < config.max_moves:
            action, root = NetworkTraining.run_mcts(config, game, network)
            game.apply(action)
            # game.see_board()
            game.store_search_statistics(root)
        return game

    # Core Monte Carlo Tree Search algorithm.
    # To decide on an action, we run N simulations, always starting at the root of
    # the search tree and traversing the tree according to the UCB formula until we
    # reach a leaf node.

    @staticmethod
    def run_mcts(config: C4Config, game: C4Game, network: Network):
        root = C4Node(0)
        NetworkTraining.evaluate(root, game, network)
        NetworkTraining.add_exploration_noise(config, root)

        for _ in range(config.num_simulations):
            node = root
            scratch_game = game.clone()
            search_path = [node]
        
        # print("Length of the search path after the for loop {}".format(len(search_path)))

        while node.expanded():
            action, node = NetworkTraining.select_child(config, node)
            scratch_game.apply(action)
            # print("node expansion and child picked {}".format(node))
            search_path.append(node)
            value = NetworkTraining.evaluate(node, scratch_game, network)
            NetworkTraining.backpropagate(
                search_path, value, scratch_game.to_play())
        # print("Length of the search path after node expansion {}".format(len(search_path)))
        return NetworkTraining.select_action(config, game, root), root

    @staticmethod
    def select_action(config: C4Config, game: C4Game, root: C4Node):
        visit_counts = [(child.visit_count, action)
                        for action, child in root.children.items()]
        
        if len(game.history) < config.num_sampling_moves:
            _, action = NetworkTraining.softmax_sample(visit_counts)
        else:
            _, action = max(visit_counts)
        return action

    # Select the child with the highest UCB score.

    @staticmethod
    def select_child(config: C4Config, node: C4Node):
        ucb_score, action, child = max((NetworkTraining.ucb_score(config, node, child), action, child)
                               for action, child in node.children.items())
        children = []
        for action, child in node.children.items():
            if (NetworkTraining.ucb_score(config, node, child) == ucb_score):
                children.append((action, child))
    

        # print("OUTPUT of max function {}".format(random.choices(children)))
        return random.choices(children)[0]

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

    # ----------------TRAINING------------------------------------------

    @staticmethod
    def train_network(config: C4Config, storage: SharedStorage,
                      replay_buffer: ReplayBuffer):
        network = Network()
        boundaries = list(config.learning_rate_schedule.keys())
        boundaries.pop(0)
        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, config.learning_rate_schedule.values())
        optimizer = tf.keras.optimizers.SGD(learning_rate_fn,
                                            config.momentum)
        while (replay_buffer.is_empty()): # sleep until there is something to do.
            time.sleep(5)
        for i in range(config.training_steps):
            print(f"At training step {i}")
            if i % config.checkpoint_interval == 0:
                print("{}At checkpoint {}/{}{}".format(BColors.OKBLUE, i, config.training_steps, BColors.ENDC))
                print("Replay buffer size: {}".format(replay_buffer.get_buffer_size()))
                storage.save_network(i, network)
            if replay_buffer.is_empty():
                continue
            batch = replay_buffer.sample_batch()
            NetworkTraining.update_weights(
                optimizer, network, batch, config.weight_decay)
        storage.save_network(config.training_steps, network)

    @staticmethod
    def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: Network, batch,
                       weight_decay: float):
        def loss_fcn():
            loss = 0
            mse = tf.keras.losses.MeanSquaredError(reduction="auto")
            for image, (target_value, target_policy) in batch:
                value, policy_logits = network.inference(image)
                target_value = [target_value]
                loss += (
                    mse(value, target_value).numpy() +
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=policy_logits, labels=target_policy))

            for weights in network.get_weights():
                loss += weight_decay * tf.nn.l2_loss(weights)

            print(loss)

            return loss
        optimizer.minimize(loss_fcn,var_list=network.get_weights())
        print(f"{BColors.OKCYAN}Finished updating weights{BColors.ENDC}")

    @staticmethod
    def softmax_sample(d):
        """
        d: list of tuples (visit_count, associated action)
        Returns softmax visit count and action.
        """
        # could add in temperature parameter if wanted
        max_visits, _ = max(d, key=lambda item: item[0])
        max_tuples = [item for item in d if item[0] == max_visits]
        return random.choice(max_tuples)

    @staticmethod
    def launch_job(f, *args):
        x = Process(target=f, args=args)
        x.start()
        return x

if __name__ == "__main__":
    network = Network()
    config = C4Config()
    game = C4Game()
    print(network.inference(game.make_image(-1)))
    # NetworkTraining.play_game(config, network)