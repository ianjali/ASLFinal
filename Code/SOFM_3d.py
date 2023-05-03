import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MiniSom:
    def __init__(self, n_neurons, m_neurons, k_neurons, input_len, sigma=1.0, learning_rate=0.5,
                 neighborhood_function='gaussian', random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        self._n = n_neurons
        self._m = m_neurons
        self._k = k_neurons
        self._input_len = input_len

        self._sigma = sigma
        self._learning_rate = learning_rate

        self._weights = np.random.rand(self._n, self._m, self._k, self._input_len) * 2 - 1

        self._neighborhood_function = self._get_neighborhood_function(neighborhood_function)

    def _get_neighborhood_function(self, neighborhood_function):
        if neighborhood_function == 'gaussian':
            return self._gaussian
        elif neighborhood_function == 'mexican_hat':
            return self._mexican_hat
        else:
            raise ValueError("Neighborhood function must be 'gaussian' or 'mexican_hat'")

    def _gaussian(self, center, sigma):
        return np.exp(-(np.square(center - np.arange(self._k)[:, None, None]) / (2 * sigma ** 2)))

    def _mexican_hat(self, center, sigma):
        d = np.abs(center - np.arange(self._k)[:, None, None])
        return (2 - (d / sigma) ** 2) * np.exp(-(d ** 2) / (2 * sigma ** 2))

    def find_winner(self, x):
        x = x.reshape((1, -1))
        dists = np.sum((self._weights - x) ** 2, axis=-1)
        winner = np.unravel_index(np.argmin(dists), self._weights.shape[:3])
        return winner

    def learning_rate(self, t):
        return self._learning_rate * np.exp(-t / (self._n * self._m))

    def neighborhood_distance(self, center):
        if self._neighborhood_function == self._gaussian:
            return self._gaussian(center, self._sigma)
        elif self._neighborhood_function == self._mexican_hat:
            return self._mexican_hat(center, self._sigma)
        else:
            raise ValueError("Neighborhood function must be 'gaussian' or 'mexican_hat'")

    def train(self, data, n_epochs):
        for epoch in range(n_epochs):
            for i, x in enumerate(data):
                winner = self.find_winner(x)
                lr = self.learning_rate(epoch * len(data) + i)
                nd = self.neighborhood_distance(winner)
                self._weights += lr * nd[:, :, :, np.newaxis] * (x - self._weights)

    def plot_weights_3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        for i in range(self._n):
            for j in range(self._m):
                for k in range(self._k):
                    ax.scatter(self._weights[i][j][k][0], self._weights[i][j][k][1], self
