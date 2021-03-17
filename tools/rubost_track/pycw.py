import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_distances


class ChineseWhispers(BaseEstimator, ClusterMixin):
    """Implementation of chinese whispers clustering algorithm.

    Source paper: https://pdfs.semanticscholar.org/c64b/9ed6a42b93a24316c7d1d6b\
                  3fddbd96dbaf5.pdf?_ga=2.16343989.248353595.1538147473-1437352660.1538147473

    """

    def __init__(self, threshold=0, n_iteration=10, metric='euclidean'):
        """Init an estimator object.

        Parameters
        ----------
        n_iterations: int
            Number of iterations.

        metric: str
            String indicating metric to use in calculating distances between samples.
            For available metrics refer to:
             http://scikit-learn.org/0.18/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html

        Attributes
        ----------
        labels_: Input data clusters will save in this attribute,
            after calling fit_predict method. This attribute
            will be a vector of shape X.shape[0] .

        adjacency_mat_: Contains pair-wise similarity measure of input data.
            Similarity measure is: 1/euclidean_distance.

        """
        self.n_iteration = n_iteration
        self.metric = metric
        self.labels_ = None
        self.adjacency_mat_ = None
        self.threshold=threshold

    def fit_predict(self, X, y=None):
        """Fits the estimator on X, and returns the labels of X as predictions.

        Parameters
        ----------
        X: :obj: np.ndarray with ndim=2
            This is the input array, rows represent data points while cols are data features.

        returns: self.labels_

        """

        assert isinstance(self.n_iteration, int), "parameter n_iterations must be of type int"
        assert isinstance(X, np.ndarray), "X must be an instance of np.ndarray"
        assert X.ndim == 2, "X must be of ndim=2"

        graph_clustering = ChineseWhispersClustering(n_iteration=self.n_iteration)
        adjacency_mat = self._generate_adjacency_mat(X,self.threshold)
        labels = graph_clustering.fit_predict(adjacency_mat)
        self.adjacency_mat_ = adjacency_mat
        self.labels_ = labels
        return labels

    def fit_predict_gm(self, X, adjacency_mat):

        assert isinstance(self.n_iteration, int), "parameter n_iterations must be of type int"
        assert isinstance(X, np.ndarray), "X must be an instance of np.ndarray"
        assert X.ndim == 2, "X must be of ndim=2"

        graph_clustering = ChineseWhispersClustering(n_iteration=self.n_iteration)
        labels = graph_clustering.fit_predict(adjacency_mat)
        self.adjacency_mat_ = adjacency_mat
        self.labels_ = labels
        return labels

    def _generate_adjacency_mat(self, X, threshold):
        n_samples = X.shape[0]
        distances_mat = pairwise_distances(X, metric=self.metric)
        adjacency_mat = (1 / (distances_mat + np.identity(n_samples, dtype=X.dtype))) *\
                        (np.ones((n_samples, n_samples), dtype=X.dtype) -
                         np.identity(n_samples, dtype=X.dtype))
        adjacency_mat[np.where(adjacency_mat <= 1/threshold)] = 0.
        return adjacency_mat

    def score(self, X, y):
        pass


class ChineseWhispersClustering:
    """ChineseWhispers algorithm.
    This class got adjancency matrix of a graph as input
    """

    def __init__(self, n_iteration=5):
        self.n_iteration = n_iteration
        self.adjacency_mat_ = None
        self.labels_ = None

    def fit_predict(self, adjacency_mat):
        """Fits and returns labels for samples"""

        n_nodes = adjacency_mat.shape[0]
        indices = np.arange(n_nodes)
        labels_mat = np.arange(n_nodes)
        for _ in range(self.n_iteration):
            np.random.shuffle(indices)
            for ind in indices:
                weights = adjacency_mat[ind]
                if len(np.where(weights > 0)[0])==0:
                    continue
                winner_label = self._find_winner_label(weights, labels_mat)
                labels_mat[ind] = winner_label
        self.adjacency_mat_ = adjacency_mat
        self.labels_ = labels_mat
        return labels_mat

    @staticmethod
    def _find_winner_label(node_weights, labels_mat):
        adjacent_nodes_indices = np.where(node_weights > 0)[0]
        adjacent_nodes_labels = labels_mat[adjacent_nodes_indices]
        unique_labels = np.unique(adjacent_nodes_labels)
        label_weights = np.zeros(len(unique_labels))
        for ind, label in enumerate(unique_labels):
            indices = np.where(adjacent_nodes_labels == label)
            weight = np.sum(node_weights[adjacent_nodes_indices[indices]])
            label_weights[ind] = weight
        winner_label = unique_labels[np.argmax(label_weights)]
        return winner_label
