import matplotlib.pyplot as plt
import numpy as np


class MultiArmedBanditEnv:
    def __init__(self, k_arms: int = 10, stationary=False):
        self.k_arms = k_arms
        self.stationary = stationary
        self.reset()

    def reset(self):
        self.q_star = np.random.randn(self.k_arms)
        self.best_action = np.argmax(self.q_star)

    def step(self, arm: int):
        if not self.stationary:
            q_drift = np.random.normal(0, 0.1, self.k_arms)
            self.q_star += q_drift
            self.best_action = np.argmax(self.q_star)

        reward: float = np.random.normal() + self.q_star[arm]
        return reward

    def render(self):
        samples = np.random.randn(1000, self.k_arms) + self.q_star
        plt.violinplot(dataset=samples, showmeans=True)
        plt.xlabel("Action")
        plt.ylabel("Reward distribution")
        plt.show()