# Reinforcement Learning for Connect Four

Goal: Train a model with reinforcement learning to master the game connect-four.

## Monte Carlo Tree Search
Developed a Monte Carlo Tree Search (based off [qpwo/monte_carlo_tree_search](https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1)) to train the reinforcement learning agent.
Strength of MCTS is indicated by the number of iterations.
One iteration:
1. Select child node to expand. A node is a game state. A child node is the next move after the parent node.
- Always pick unexplored nodes first.
- If all children of parent node are expanded, descend a layer by picking the node with the greatest upper bound for trees. Let exploration weight be sqrt(2).

```
def uct(n):
    "Upper confidence bound for trees"
    return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
        log_N_vertex / self.N[n]
    )
```

2. Expand the node and store in tree.
3. Simulate game play by picking subsequent random children. Calculate end of game reward (1 if won, 0.5 if tied, and 0 if lost).
4. Back-propagate results up the tree. 
- Update number of visits to each node.
- Add rewards to each node, inverting if the node represents an opponent's move.
5. Select the node with the highest average reward.