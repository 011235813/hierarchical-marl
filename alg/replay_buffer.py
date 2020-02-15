import random
import numpy as np

class Replay_Buffer():

    def __init__(self, size=1e6):
        self.memory = []
        self.maxsize = int(size)
        self.idx = 0

    def add(self, transition):
        if self.idx >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.idx] = transition
        self.idx = (self.idx + 1) % self.maxsize
    
    def sample(self, length):
        """
        Randomly samples one episode, and samples a subsequence
        of <length> from episode
        """
        episode = random.choice(self.memory)
        if len(episode) <= length:
            return episode
        else:
            start = np.random.randint(0, len(episode)+1-length)
            subsequence = episode[start : start + length]
            return subsequence

    def sample_batch(self, batch_size):
        """
        Samples a batch of transitions.
        Assumes that each entry in the replay buffer is a 
        single transition
        """
        if len(self.memory) <= batch_size:
            return np.array(self.memory)
        else:
            return np.array(random.sample(self.memory, batch_size))
