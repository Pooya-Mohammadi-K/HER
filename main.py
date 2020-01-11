import numpy as np

from environment import Environment
from agent import Agent
from her import HER
from utils import plot, LRScheduler

size = 20
episodes = 35000
gamma = 0.99
tau = 0.005
memory_size = 100000
epsilon_high = 0.9
epsilon_low = 0.1
epsilon_decay = 2000
lr = 0.0005
batch_size = 32
moving_coefficient = 0.01
her_parameters = {"type_": 'future', "k": 4}
# her_parameters = {"type_": 'final', }
lr_scheduler = LRScheduler(episodes=[15000, 20000, 30000], learning_rates=[0.0001, 0.00005, 0.00001], default_lr=lr)

if __name__ == '__main__':
    env = Environment(size)
    agent = Agent(size,
                  gamma,
                  tau,
                  memory_size,
                  epsilon_high,
                  epsilon_low,
                  epsilon_decay,
                  lr, batch_size)
    her = HER(size)
    moving_success_rate = [0]
    moving_reward = [0]
    moving_loss = [0]
    for episode in range(episodes):
        s, done = env.reset()
        her.reset()
        episode_loss = 0
        episode_reward = 0
        for step in range(size):
            a = agent.choose_action(s)
            s_, r, d = env.step(s, a)
            transition = [s, a, r, s_, d]
            agent.store_transition(transition)
            her.buffer.append(transition)
            # when done is true what ever her does for final is repetition
            loss = agent.train()
            episode_loss += loss
            episode_reward += r
            s = s_
            if d:
                break
        moving_success_rate.append((1 - moving_coefficient) * moving_success_rate[-1] + moving_coefficient * int(d))
        moving_loss.append((1 - moving_coefficient) * moving_loss[-1] + moving_coefficient * int(episode_loss))
        moving_reward.append((1 - moving_coefficient) * moving_reward[-1] + moving_coefficient * int(episode_reward))

        her_transition = her.back_ward(**her_parameters)
        for transition in her_transition:
            agent.store_transition(transition)
        lr = lr_scheduler(episode, agent.optimizer)
        print('episode: ', episode, 'step: ', step, 'total_loss', np.round(episode_loss, 2), 'total_reward ',
              episode_reward, 'epsilon: ', np.round(agent.epsilon, 2), 'learning_rate: ', lr)
    plot(moving_success_rate, title='success_rate', y_label='moving success_rate')
    plot(moving_reward, title='moving reward', y_label='moving reward')
    plot(moving_loss, title='moving loss', y_label='moving loss', show=True)
