from typing import Optional
import numpy as np
from numpy import typing as nptype
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)


class WindyGridWorld:
    COLS = 10
    ROWS = 7
    START = 1
    START_POINT = [3, 0]
    FINISH = 2
    FINISH_POINT = [3, 7]

    def __init__(self):
        """
        file_path: can be relative or absolute path to csv file
        """
        self.grid_map = np.zeros((self.ROWS, self.COLS))
        self.grid_map[tuple(self.START_POINT)] = self.START
        self.grid_map[tuple(self.FINISH_POINT)] = self.FINISH
        self.vertical_winds = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        self.action_space = np.random.permutation(
            [[i, j] for i in range(-1, 2) for j in range(-1, 2)],
        )
        self.reset()

    def reset(self):
        self.current_state = np.array(self.START_POINT)

    def step(self, action: nptype.NDArray[np.int32]):
        next_state = self.current_state + action
        next_state[0] -= self.vertical_winds[self.current_state[1]]
        next_state = np.clip(next_state, 0, [self.ROWS - 1, self.COLS - 1])

        self.current_state = next_state
        if self.grid_map[tuple(next_state)] == self.FINISH:
            return -1, True
        else:
            return -1, False

    def render(self, state_history: Optional[list[np.ndarray]] = None):
        plt.figure(figsize=(8, 4))
        ax = sns.heatmap(
            self.grid_map,
            cbar=False,
            square=True,
            cmap=["white", "pink", "lightgreen"],
            linewidths=0.5,  # type:ignore
            linecolor="lightgray",
        )
        plt.axline(
            (0, self.ROWS),
            (self.COLS, self.ROWS),
            color="lightgray",
        )
        plt.axline(
            (self.COLS, 0),
            (self.COLS, self.ROWS),
            color="lightgray",
        )
        if state_history:
            xs = [state[1] + 0.5 for state in state_history]
            ys = [state[0] + 0.5 for state in state_history]
            ax.plot(xs, ys, marker="o")


def sample_policy_episode(
    env: WindyGridWorld, policy: nptype.NDArray[np.int32], max_time: int
):
    env.reset()
    state_history = [env.current_state]
    for iter in range(max_time):
        state = tuple(env.current_state)
        action = policy[state]
        reward, finished = env.step(env.action_space[action])
        state_history.append(env.current_state)
        if finished:
            break
    env.render(state_history)
