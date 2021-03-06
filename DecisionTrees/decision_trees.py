import time
import numpy as np


def cum_sse(y):
    return (np.cumsum(y**2) - (np.cumsum(y))**2/np.arange(1, len(y) + 1))[:-1]


def cum_sse_mat(Y):
    return (np.cumsum(Y**2, axis=0) - (np.cumsum(Y, axis=0) ** 2) /
            np.arange(1, len(Y) + 1).reshape(-1, 1))[:-1]


def variance(y):
    return np.mean((y - np.mean(y))**2)


def split_col(x, y):
    ys = y[np.argsort(x)]
    xs = x[np.argsort(x)]
    sse = cum_sse(ys) + cum_sse(ys[::-1])[::-1]
    m_sse = np.min(sse)
    x_split = (xs[np.argmin(sse)] + xs[np.argmin(sse) + 1])/2.
    return variance(y) - m_sse/len(y), x_split


def split_matrix(X, y):
    n_obs, n_feat = X.shape
    m_sse = np.zeros(n_feat)
    x_s = m_sse.copy()
    for i in range(n_feat):
        m_sse[i], x_s[i] = split_col(X[:, i], y)
    return np.argmax(m_sse), x_s[np.argmax(m_sse)], np.max(m_sse)


def split_matrix_fast(X, y):
    Y_sorted = y.reshape(-1, 1)[np.argsort(X, axis=0),
                                np.zeros(X.shape, dtype=int)]
    sse = cum_sse_mat(Y_sorted) + cum_sse_mat(Y_sorted[::-1])[::-1]
    best_col = np.argmin(np.min(sse, axis=0))
    impurity = variance(y) - np.min(sse[:, best_col]) / len(X)
    split_idx = np.argmin(sse[:, best_col])
    split_value = np.mean(np.sort(X[:, best_col])[split_idx:(split_idx + 2)])
    return best_col, split_value, impurity


class DecisionTrees():
    def __init__(self):
        self.depth = 10
        self._nodes = np.zeros(2**self.depth, dtype=[('left_child', np.int64),
                     ('right_child', np.int64), ('feature', np.int64),
                     ('threshold', np.float64), ('impurity', np.float64),
                     ('n_node_samples', np.int64)])
        self.n_nodes = 0

    def add_node(self, p_index, feature, threshold, impurity, n_nodes_sample):
        if self.n_nodes >= len(self._nodes):
            self.resize_nodes()
        self._nodes[index] = (-1, -1, feature, threshold,
                              impurity, n_nodes_sample)

class DecisionTreesRegressor():
    def __init__(self, max_depth):
        self.max_depth = max_depth
    def fit(self, X, y):
        self.tree = np.zeros((2**max_depth-1, 6))

if __name__ == '__main__':
    # X = np.array([[1, 2, 3, 4, 5], [3, 2, 5, 1, 7]]).T
    # y = np.array([0, 0, 1, 1, 1])
    n = 10000
    p = 100
    X = np.random.randn(n, p)
    beta = np.arange(p)
    signal = X.dot(beta)
    y = signal + np.random.randn(n) * np.std(signal)

    a = time.time()
    print(split_matrix(X, y))
    b = time.time()
    print(split_matrix_fast(X, y))
    c = time.time()
    print("Bui Nghia     : ", b - a)
    print("Speed of light: ", c - b)
