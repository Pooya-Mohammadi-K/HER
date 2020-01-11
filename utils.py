import matplotlib.pyplot as plt


def plot(lst, title='', y_label='', x_label='', show=False):
    plt.figure()
    plt.plot(lst, 'b--')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if show:
        plt.show()


class LRScheduler:
    def __init__(self, episodes, learning_rates, default_lr):
        self.episodes = episodes
        self.learning_rates = learning_rates
        self.default_lr = default_lr

    def __call__(self, episode, optimizer):
        lr = self.default_lr
        for i, episode_ in enumerate(self.episodes):
            if episode > episode_:
                lr = self.learning_rates[i]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr
