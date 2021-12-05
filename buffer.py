import numpy as np


class Buffer:
    def __init__(self, max_size=500000):
        self.buffer = []
        self.max_size = max_size
        self.size = 0

    def add(self, x):
        self.size += 1
        if self.size < self.max_size:
            self.buffer.append(x)
        else:
            self.buffer[self.size % self.max_size] = x

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        samples = [self.buffer[i] for i in indexes]
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
