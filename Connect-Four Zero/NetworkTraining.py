#!/usr/bin/env python
from C4Config import C4Config
from SharedStorage import SharedStorage
from ReplayBuffer import ReplayBuffer
from Network import Network
from C4Game import C4Game
from C4Node import C4Node

import math
from typing import List
import numpy as np
import tensorflow as tf


def alphazero(config: C4Config):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    for i in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()


# ----------------SELF PLAY----------------------------------------

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: C4Config, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: C4Config, network: Network):
    game = C4Game()
    while not game.terminal() and len(game.history) < config.max_moves:
        action, root = run_mcts(config, game, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: C4Config, game: C4Game, network: Network):
    root = C4Node(0)
    evaluate(root, game, network)
    add_exploration_noise(config, root)

    for _ in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

    while node.expanded():
        action, node = select_child(config, node)
        scratch_game.apply(action)
        search_path.append(node)

        value = evaluate(node, scratch_game, network)
        backpropagate(search_path, value, scratch_game.to_play())
    return select_action(config, game, root), root


def select_action(config: C4Config, game: C4Game, root: C4Node):
    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.iteritems()]
    if len(game.history) < config.num_sampling_moves:
        _, action = softmax_sample(visit_counts)
    else:
        _, action = max(visit_counts)
    return action


# Select the child with the highest UCB score.
def select_child(config: C4Config, node: C4Node):
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in node.children.iteritems())
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: C4Config, parent: C4Node, child: C4Node):
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: C4Node, game: C4Game, network: Network):
    value, policy_logits = network.inference(game.make_image(-1))

    # Expand the node.
    node.to_play = game.to_play()
    policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
    policy_sum = sum(policy.itervalues())
    for action, p in policy.iteritems():
        node.children[action] = C4Node(p / policy_sum)
    return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[C4Node], value: float, to_play):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else (1 - value)
        node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: C4Config, node: C4Node):
    actions = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


# ----------------TRAINING------------------------------------------


def train_network(config: C4Config, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = Network()
    optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
                                           config.momentum)
    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch()
        update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(config.training_steps, network)


def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
                   weight_decay: float):
    loss = 0
    for image, (target_value, target_policy) in batch:
        value, policy_logits = network.inference(image)
        loss += (
            tf.losses.mean_squared_error(value, target_value) +
            tf.nn.softmax_cross_entropy_with_logits(
                logits=policy_logits, labels=target_policy))

    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    optimizer.minimize(loss)


def softmax_sample(d):
    return 0, 0


def launch_job(f, *args):
    f(*args)




#model training code from NeuralNetworkTrainer that I thought be useful 
# def __train_model(self, model):
#         # trains model for the ith iteration
#         # how to use train_labels if there are 2 of them?
#         model.fit(..., ..., epochs=80, validation_data=..., callbacks=...)

#     def iteratively_train_model(self, iterations: int, num_self_play_games: int, MCTS_iterations: int):
#         """
#         iterations: number of iterations to train model
#         num_self_play_games: number of self-play games to play each iteration i
#         MCTS_iterations: number of iterations in MCTS before picking a move for a self-play game

#         Trains a deep convolution neural network for connect-4 iteratively using self-play games and MCTS.
#         Neural net outputs move probabilities and predicted game winner from perspective of current player.
#         Needs no external training data; that will be generated by self-play games.

#         TODO: implement
#         """
#         for _ in range(iterations):
#             for _ in range(num_self_play_games):
#                 self.__play_self_play_game(MCTS_iterations)
#             model = NeuralNetworkTrainer.get_model()
#             model.load_weights(...)
#             self.__train_model(model)