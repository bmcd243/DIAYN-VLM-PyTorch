import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'z', 'done', 'action', 'next_state', 'vlm_emb'))


class Memory:
    def __init__(self, buffer_size, seed):
        self.buffer_size = buffer_size
        self.buffer = []
        self.seed = seed
        random.seed(self.seed)

    def add(self, state, z, done, action, next_state, vlm_emb):
        self.buffer.append(Transition(state, z, done, action, next_state, vlm_emb))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        assert len(self.buffer) <= self.buffer_size

    def sample(self, size):
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)

    @staticmethod
    def get_rng_state():
        return random.getstate()

    @staticmethod
    def set_rng_state(random_rng_state):
        random.setstate(random_rng_state)
