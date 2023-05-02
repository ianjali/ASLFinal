import pandas as pd
import numpy as np
#from minisom import MiniSom
import matplotlib.pyplot as plt


class MiniSomImplement:
    def __init__(self, n_neurons, m_neurons, input_len, sigma=1.0, learning_rate=0.5,
                 neighborhood_function='mexican_hat', random_seed=None):
        self.n_neurons = n_neurons
        self.m_neurons = m_neurons
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.neighborhood_function = neighborhood_function
        if random_seed is not None:
            np.random.seed(random_seed)
        self.weights = np.random.rand(n_neurons, m_neurons, input_len)
        #self.neighbors = self._find_neighbors()

    def train(self, data, n_iterations):
        for i in range(n_iterations):
            rand_i = np.random.randint(len(data))
            input_vect = data[rand_i]
            bmu = self.find_winner(input_vect)
            self._update_weights(input_vect, bmu, i, n_iterations)

    def find_winner(self, input_vect):
        """
        Find the winner neuron for the input_vect.
        """
        min_dist = np.inf
        winner = None
        for i in range(self.n_neurons):
            for j in range(self.m_neurons):
                weight_vect = self.weights[i, j, :]
                dist = np.linalg.norm(input_vect - weight_vect)
                if dist < min_dist:
                    min_dist = dist
                    winner = (i, j)
        return winner

    def cal_learning_rate(self, t, max_iter):
        """
        Calculate the learning rate as a function of the current iteration number t and the maximum number of iterations max_iter.
        """
        return self.learning_rate * (1 - t / max_iter)

    def neighborhood_distance(self, c1, c2):
        """
        Calculate the Euclidean distance between two neurons with coordinates c1 and c2.
        """
        return np.linalg.norm(np.array(c1) - np.array(c2))

    def mexican_hat(self, d, sigma):
        """
        Calculate the value of the Mexican hat function for a given distance d and sigma parameter.
        """
        return (2 / np.sqrt(3 * sigma)) * (1 - ((d ** 2) / (sigma ** 2))) * np.exp(-((d ** 2) / (2 * (sigma ** 2))))

    def _update_weights(self, input_vect, bmu, t, max_iter):
        """
        Update the weights of the BMU and its neighbors based on the input_vect and the current iteration number t.
        """
        for i in range(self.n_neurons):
            for j in range(self.m_neurons):
                weight_vect = self.weights[i, j, :]
                dist_to_bmu = self.neighborhood_distance((i, j), bmu)
                if self.neighborhood_function == 'gaussian':
                    neighbor_strength = np.exp(-(dist_to_bmu ** 2) / (2 * (self.sigma ** 2)))
                elif self.neighborhood_function == 'mexican_hat':
                    neighbor_strength = self.mexican_hat(dist_to_bmu, self.sigma)
                else:
                    raise ValueError('Invalid neighborhood function')
                learning_rate = self.cal_learning_rate(t, max_iter)
                delta = learning_rate * neighbor_strength * (input_vect - weight_vect)
                self.weights[i, j, :] += delta

    def plot_weights(weights, targets):
        n_neurons, m_neurons, input_len = weights.shape
        plt.figure(figsize=(10, 10))
        for i in range(n_neurons):
            for j in range(m_neurons):
                plt.subplot(n_neurons, m_neurons, i * m_neurons + j + 1)
                plt.imshow(weights[i, j, :].reshape((int(np.sqrt(input_len)), -1)))
                plt.xticks([])
                plt.yticks([])
                plt.title(targets[i * m_neurons + j])
        plt.tight_layout()
        plt.show()

    def _find_neighbors(self):
        """
        Find the neighboring neurons of each neuron on the SOM grid.
        """
        neighbors = {}
        for i, neuron in enumerate(self.neurons):
            dists = np.array([self.neighborhood_distance(neuron, other) for other in self.neurons])
            neighbors[neuron] = np.argsort(dists)[1:]
        return neighbors

    def distance_map(self):
        """
        Returns the distance map of the SOM.
        """
        # Calculate the distance between each neuron and its neighbors.
        distances = np.zeros((self.n_neurons, self.m_neurons))
        for i in range(self.n_neurons):
            for j in range(self.m_neurons):
                for k in range(self.n_neurons):
                    for l in range(self.m_neurons):
                        distances[i, j] += np.linalg.norm(self.weights[i, j, :] - self.weights[k, l, :])

        # Calculate the average distance between each neuron and its neighbors.
        avg_distances = np.zeros((self.n_neurons, self.m_neurons))
        for i in range(self.n_neurons):
            for j in range(self.m_neurons):
                neighbors = self.neighbours((i, j))
                avg_distances[i, j] = np.mean([distances[k, l] for k, l in neighbors])

        # Create a matrix with the average distances.
        dist_map = np.zeros((self.n_neurons, self.m_neurons))
        for i in range(self.n_neurons):
            for j in range(self.m_neurons):
                dist_map[i, j] = avg_distances[i, j]

        # Normalize the matrix.
        dist_map = dist_map / dist_map.max()

        return dist_map

    def plot_weights_target(self,targets,data):
        # w_x, w_y = zip(*[self.find_winner(d) for d in data])
        # w_x = np.array(w_x)
        # w_y = np.array(w_y)
        #
        # plt.figure(figsize=(10, 9))
        # plt.pcolor(self.distance_map().T, cmap='bone_r', alpha=.2)
        # plt.colorbar()
        plt.figure(figsize=(16, 16))
        for i, (x, t) in enumerate(zip(data, targets)):
            w = self.find_winner(x)#find_winner
            #print(w)
            plt.text(w[0] + .5, w[1] + .5, str(t), color=plt.cm.tab20(t / 10.), fontdict={'weight': 'bold', 'size': 11})
        plt.axis([0, self.n_neurons, 0, self.m_neurons])
        plt.title('SOM Clusters')
        plt.show()
    def plot_weights_3d(self,targets,data):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i, (x, t) in enumerate(zip(data, targets)):
            w = self.find_winner(x)
            ax.scatter(w[0], w[1], t, color=plt.cm.tab20(t / 10.), s=100, alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Encoded Label')
        ax.set_title('SOM Clusters in 3D')
        plt.show()

    def adjacentt(self, coord):
        """
        Find the adjacent neurons to the neuron with coordinates coord.
        """
        x, y = coord
        adj_neurons = []
        for i in range(self.n_neurons):
            for j in range(self.m_neurons):
                if self.distance_map[i, j] == self.distance_map[x, y] - 1:
                    adj_neurons.append((i, j))
        return adj_neurons

    def quantization_error(som, data):
        q_error = 0
        for vector in data:
            bmu = som.find_winner(vector)
            q_error += np.linalg.norm(vector - som.weights[bmu])
        return q_error  / len(data)

    def topographic_error(som, data):
        t_error = 0
        for vector in data:
            bmu, second_bmu = som.find_winners(vector)
            if not som._is_adjacent(bmu, second_bmu):
                t_error += 1
        return t_error / len(data)

# class MiniSom:
#     def __init__(self, n_neurons, m_neurons, input_dim, sigma=1.0, learning_rate=0.5,
#                  neighborhood_function='gaussian', random_seed=0):
#         self.n_neurons = n_neurons
#         self.m_neurons = m_neurons
#         self.input_dim = input_dim
#         self.sigma = sigma
#         self.learning_rate = learning_rate
#         self.neighborhood_function = neighborhood_function
#         self.random_seed = random_seed
#         self.weights = np.random.rand(n_neurons, m_neurons, input_dim)
#
#     def train(self, data, num_epochs):
#         np.random.seed(self.random_seed)
#         for epoch in range(num_epochs):
#             for i in range(data.shape[0]):
#                 input_vec = data[i]
#                 winner = self._find_winner(input_vec)
#                 self._update_weights(winner, input_vec, epoch, num_epochs)
#
#     def _find_winner(self, input_vec):
#         distances = np.linalg.norm(self.weights - input_vec, axis=2)
#         return np.unravel_index(distances.argmin(), distances.shape)
#
#     def _update_weights(self, winner, input_vec, epoch, num_epochs):
#         alpha = self._learning_rate(epoch, num_epochs)
#         h = self._neighborhood_function(winner, epoch, num_epochs)
#         for i in range(self.n_neurons):
#             for j in range(self.m_neurons):
#                 self.weights[i][j] += alpha * h[i][j] * (input_vec - self.weights[i][j])
#
#     def _learning_rate(self, epoch, num_epochs):
#         return self.learning_rate * np.exp(-epoch / num_epochs)
#
#     def _neighborhood_function(self, winner, epoch, num_epochs):
#         if self.neighborhood_function == 'gaussian':
#             return np.exp(-self._distance(winner) ** 2 / (2 * self.sigma ** 2))
#         elif self.neighborhood_function == 'mexican_hat':
#             return self._mexican_hat(winner)
#         else:
#             raise ValueError('Invalid neighborhood function specified')
#
#     def _distance(self, neuron):
#         return np.sqrt((np.arange(self.n_neurons) - neuron[0]) ** 2 +
#                        (np.arange(self.m_neurons)[:, np.newaxis] - neuron[1]) ** 2)
#
#     def _mexican_hat(self, neuron):
#         d = self._distance(neuron)
#         return (1 - d ** 2) * np.exp(-d ** 2 / (2 * self.sigma ** 2))

# from sklearn.cluster import KMeans
# def k_means(vecs, nclusters):
#     kmeans = KMeans(n_clusters=nclusters, n_init=10, n_jobs=2)
#     kmeans.fit(vecs)
#     return kmeans