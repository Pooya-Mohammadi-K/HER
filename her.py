from copy import deepcopy as dc
import random


class HER:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.transition_buffer = []

    def reset(self):
        self.buffer.clear()
        self.transition_buffer.clear()

    def back_ward(self, type_, **kwargs):
        if type_ == 'future':
            k = kwargs['k']
            self.k_future(k)
        else:
            self.final()
        return dc(self.transition_buffer)

    def final(self):
        _, _, _, s_, _ = self.buffer[-1]
        target = s_[:self.size]
        for transition in self.buffer:
            s, a, r, s_, d = transition
            s_[self.size:] = target
            s[self.size:] = target
            dist = (s_[:self.size] - s_[self.size:]).abs().sum()
            if dist == 0:
                r = 0
                d = True
            else:
                r = -1
                d = False
            self.transition_buffer.append([s, a, r, s_, d])

    def k_future(self, k):
        for i, transition in enumerate(self.buffer):
            s, a, r, s_, d = transition
            k_next = random.choices(self.buffer[i:], k=k)  # how to sample with replacement in python...
            for next_transition in k_next:
                _, _, _, next_s_, _ = next_transition
                target = next_s_[:self.size]
                s_[self.size:] = target
                s[self.size:] = target
                dist = (s_[:self.size] - s_[self.size:]).abs().sum()
                if dist == 0:
                    r = 0
                    d = True
                else:
                    r = -1
                    d = False
                self.transition_buffer.append([s, a, r, s_, d])
