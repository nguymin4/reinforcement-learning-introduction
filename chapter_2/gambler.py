import numpy as np


class Gambler:
    def arm(self):
        raise NotImplementedError()

    def update(self, action: int, reward: float):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class AveragingGambler(Gambler):
    def __init__(self, total_actions: int = 10, epsilon: float = 0):
        self.total_actions = total_actions
        self.epsilon = epsilon
        self.reset()

    def act(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.total_actions)
        return np.argmax(self.Q)

    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def reset(self):
        self.N = np.zeros(self.total_actions)
        self.Q = np.zeros(self.total_actions)


class FixedLearningStepGambler(Gambler):
    def __init__(self, total_actions: int = 10, epsilon: float = 0, alpha=0.1):
        self.total_actions = total_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.reset()

    def act(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.total_actions)
        return np.argmax(self.Q)

    def update(self, action: int, reward: float):
        self.Q[action] += (reward - self.Q[action]) * self.alpha

    def reset(self):
        self.Q = np.zeros(self.total_actions)


class UCBGambler(Gambler):
    def __init__(self, total_actions: int = 10, exploration_degree: float = 1, **_kwargs):
        self.total_actions = total_actions
        self.exploration_degree = exploration_degree
        self.reset()

    def act(self):
        ucb = self.Q + self.exploration_degree * np.sqrt(np.log(self.t) / (self.N + 1e-8))
        return np.argmax(ucb)

    def update(self, action: int, reward: float):
        self.t += 1
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def reset(self):
        self.t = 1
        self.N = np.zeros(self.total_actions)
        self.Q = np.zeros(self.total_actions)