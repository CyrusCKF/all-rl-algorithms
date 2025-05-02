import pathlib
import random

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from ray import tune, train
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module import RLModule, MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.examples._old_api_stack.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec


class TicTacToe(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()

        self.agents = self.possible_agents = ["player1", "player2"]

        self.observation_spaces = {
            "player1": Box(-1.0, 1.0, (9,), np.float32),
            "player2": Box(-1.0, 1.0, (9,), np.float32),
        }

        self.action_spaces = {
            "player1": Discrete(9),
            "player2": Discrete(9),
        }

        self.board = None
        self.current_player = None

    def reset(self, *, seed=None, options=None):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.current_player = str(np.random.choice(["player1", "player2"]))

        return {self.current_player: np.array(self.board, np.float32)}, {}

    def step(self, action_dict):
        action = action_dict[self.current_player]

        rewards = {self.current_player: 0.0}
        terminateds = {"__all__": False}

        opponent = "player1" if self.current_player == "player2" else "player2"

        # Penalize trying to place a piece on an already occupied field.
        if self.board[action] != 0:
            rewards[self.current_player] -= 1.0
            terminateds["__all__"] = True
        # Change the board according to the (valid) action taken.
        else:
            was = self.winning_actions()
            wfo = self.winning_actions(for_opponent=True)

            # Penalize if player has a winning action option but doesn't choose it
            if was and action not in was:
                rewards[self.current_player] -= 1.0
                # terminateds["__all__"] = True  # todo Only terminate in training, not in evaluation

            # Penalize if player doesn't prevent opponent winning in the next round
            elif wfo and action not in wfo:
                rewards[self.current_player] -= 1.0
                # terminateds["__all__"] = True # todo Only terminate in training, not in evaluation

            self.board[action] = 1 if self.current_player == "player1" else -1

            # After having placed a new piece, figure out whether the current player
            # won or not.
            if self.current_player == "player1":
                win_val = [1, 1, 1]
            else:
                win_val = [-1, -1, -1]
            if (
                # Horizontal
                self.board[0:3] == win_val
                or self.board[3:6] == win_val
                or self.board[6:9] == win_val
                or
                # Vertical
                [self.board[i] for i in [0, 3, 6]] == win_val
                or [self.board[i] for i in [1, 4, 7]] == win_val
                or [self.board[i] for i in [2, 5, 8]] == win_val
                or
                # Diagonal
                [self.board[i] for i in [0, 4, 8]] == win_val
                or [self.board[i] for i in [2, 4, 6]] == win_val
            ):
                # Final reward is for victory and loss
                rewards[self.current_player] += 10.0
                rewards[opponent] = -10.0

                terminateds["__all__"] = True

            # The board might also be full w/o any player having won/lost.
            # In this case, we simply end the episode and the players get some reward for a tie
            elif 0 not in self.board:
                terminateds["__all__"] = True
                rewards[self.current_player] += 1.0
                rewards[opponent] = 1.0

        # Flip players and return an observations dict with only the next player to
        # make a move in it.
        self.current_player = opponent

        obss = {self.current_player: np.array(self.board, np.float32)}

        return (
            obss,
            rewards,
            terminateds,
            {},
            {},
        )

    # Return all actions that result into the player (or opponent) winning
    def winning_actions(self, for_opponent=False):
        opponent = "player1" if self.current_player == "player2" else "player2"

        player = self.current_player if not for_opponent else opponent

        winning_actions = []

        for action in range(0, 9):
            if self.board[action] != 0:
                continue

            b = self.board.copy()
            b[action] = 1 if player == "player1" else -1

            if player == "player1":
                win_val = [1, 1, 1]
            else:
                win_val = [-1, -1, -1]

            if (
                # Horizontal
                b[0:3] == win_val
                or b[3:6] == win_val
                or b[6:9] == win_val
                or
                # Vertical
                [b[i] for i in [0, 3, 6]] == win_val
                or [b[i] for i in [1, 4, 7]] == win_val
                or [b[i] for i in [2, 5, 8]] == win_val
                or
                # Diagonal
                [b[i] for i in [0, 4, 8]] == win_val
                or [b[i] for i in [2, 4, 6]] == win_val
            ):
                winning_actions.append(action)

        return winning_actions


def print_obs(obs):
    if not obs:
        return

    print("obs:")

    current_player = list(obs.keys())[0]
    l1 = [int(i) for i in list(obs[current_player])]
    print(l1[0:3])
    print(l1[3:6])
    print(l1[6:9])


env = TicTacToe()

config = (
    PPOConfig()
    .environment(TicTacToe)
    .env_runners(num_env_runners=1, batch_mode="complete_episodes")
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={"player1": RLModuleSpec()},
        ),
    )
    .multi_agent(
        policies={
            "player1": PolicySpec(),
            "player2": PolicySpec(
                policy_class=RandomPolicy,
                observation_space=Box(-1.0, 1.0, (9,), np.float32),
                action_space=Discrete(9),
            ),
        },
        policy_mapping_fn=lambda aid, *a, **kw: aid,
        policies_to_train=["player1"],
    )
    .training(
        lr=tune.grid_search([0.01, 0.001, 0.0001, 0.00001]),
        train_batch_size_per_learner=4000,
        minibatch_size=1000,
        num_epochs=5,
        gamma=0.99,
        entropy_coeff=0.01,
        clip_param=0.2,
    )
)

best_checkpoint_path = ""

if not best_checkpoint_path:
    print("Init tuner")
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=train.RunConfig(
            stop={"num_env_steps_sampled_lifetime": 200000},
            checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True),
        ),
    )
    print("Fit tuner")
    results = tuner.fit()

    print("Get best result")

    best_result = results.get_best_result(
        metric="env_runners/agent_episode_returns_mean/player1", mode="max"
    )
    best_checkpoint_path = best_result.checkpoint.path

    print("Get best checkpoint")
    best_checkpoint = best_result.checkpoint
    best_checkpoint_path = best_checkpoint.path
    print(f"best_checkpoint_path = {best_checkpoint_path}")


print("Get RL Module from checkpoint")
rl_module = RLModule.from_checkpoint(
    pathlib.Path(best_checkpoint_path) / "learner_group" / "learner" / "rl_module"
)["player1"]

print("Start evalution")
wins = ties = losses = 0
total = 10000
for _ in range(total):
    episode_return = 0
    terminated = {"__all__": False}
    episode_return_player1 = episode_return_player2 = 0

    obs, info = env.reset()

    last_reward1 = last_reward2 = None
    while not terminated["__all__"]:
        current_player = list(obs.keys())[0]

        if current_player == "player1":
            torch_obs_batch = torch.from_numpy(np.array([obs[current_player]]))

            action_logits = rl_module.forward_inference({"obs": torch_obs_batch})[
                "action_dist_inputs"
            ]

            sorted_indices = torch.argsort(action_logits, descending=True).tolist()[0]
            sorted_tensor, _ = torch.sort(action_logits, descending=True)
            sorted_list = sorted_tensor.squeeze().tolist()
            sorted_list_rounded = [
                round(num, 3) for num in sorted_tensor.squeeze().tolist()
            ]
        else:  # player2 plays random, legal moves
            sorted_indices = random_list = random.sample(range(9), 9)
            sorted_list_rounded = [round(num, 3) for num in [1 / 9] * 9]

        i = 0
        while i == 0 or int(obs[current_player][action]) != 0:
            action = sorted_indices[i]
            i += 1

        action_dict = {current_player: action}
        obs, reward, terminated, truncated, info = env.step(action_dict)

        # For debugging the exact moves:
        # print(f'{current_player}: action = {int(action)}, {sorted_indices_list}')
        # print(f'weights: {sorted_list_rounded}')
        # print_obs(obs)
        # print(f'reward: {reward}')
        # print()

        episode_return_player1 += reward["player1"] if "player1" in reward else 0
        episode_return_player2 += reward["player2"] if "player2" in reward else 0
        last_reward1 = reward["player1"] if "player1" in reward else -1
        last_reward2 = reward["player2"] if "player2" in reward else -1

    if last_reward1 > last_reward2:
        wins += 1
    elif last_reward1 == last_reward2:
        ties += 1
    else:
        losses += 1

print(f"wins: {wins}, ties: {ties}, losses:{losses}, total: {total}")
