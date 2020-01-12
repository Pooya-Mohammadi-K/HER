from copy import deepcopy as dc
from collections import deque
import random

import numpy as np


class HER:
    def __init__(self, size, rewards):
        self.size = size
        self.buffer = []
        self.transition_buffer = []
        self.rewards = rewards

    def reset(self):
        self.buffer.clear()
        self.transition_buffer.clear()

    def back_ward(self, type_, **kwargs):
        if type_ == 'future':
            k = kwargs['k']
            self.k_future(k)
        elif type_ == 'n_step_final':
            n = kwargs['n']
            gamma = kwargs['gamma']
            self.n_step_final(n, gamma)
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
                r = self.rewards[0]
                d = True
            else:
                r = self.rewards[1]
                d = False
            self.transition_buffer.append([s, a, r, s_, d])

    def n_step_final(self, n, gamma):
        n_step_buffer = []
        _, _, _, s_, _ = self.buffer[-1]
        target = s_[:self.size]

        for transition in self.buffer:
            s, a, r, s_, d = transition
            s_[self.size:] = target
            s[self.size:] = target
            dist = (s_[:self.size] - s_[self.size:]).abs().sum()
            if dist == 0:
                r = self.rewards[0]
                d = True
            else:
                r = self.rewards[1]
                d = False
            n_step_buffer.append([s, a, r, s_, d])
            if len(n_step_buffer) >= n:
                n_return, d = self.get_n_return(gamma, n, n_step_buffer)
                s, a, r, s_, state_d = n_step_buffer.pop(0)
                self.transition_buffer.append([s, a, n_return, s_, d])
        while len(n_step_buffer):
            n_return, _ = self.get_n_return(gamma, n, n_step_buffer)
            s, a, _, _, _ = n_step_buffer.pop(0)
            self.transition_buffer.append([s, a, n_return, s_, True])

        v = 10

    @staticmethod
    def get_n_return(gamma, n, n_step_buffer):
        n_return = 0
        done = False
        for i in range(n):
            _, _, r_, _, current_d = n_step_buffer[i]
            if current_d:
                n_return += r_ * (gamma ** i)
                done = True
                break
            n_return += r_ * (gamma ** i)
        return n_return, done

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
                    r = self.rewards[0]
                    d = True
                else:
                    r = self.rewards[1]
                    d = False
                self.transition_buffer.append([s, a, r, s_, d])
