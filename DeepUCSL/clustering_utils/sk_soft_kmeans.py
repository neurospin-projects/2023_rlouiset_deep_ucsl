import torch
import numpy as np
from DeepUCSL.clustering_utils.sinkhorn_knopp import distributed_sinkhorn

@torch.no_grad()
class SoftSKKMeans:
    '''
    Kmeans clustering algorithm implemented with PyTorch
    Parameters:
      n_clusters: int,
        Number of clusters
      max_iter: int, default: 100
        Maximum number of iterations
      tol: float, default: 0.0001
        Tolerance

      mode: {'euclidean', 'cosine'}, default: 'euclidean'
        Type of distance measure
      minibatch: {None, int}, default: None
        Batch size of MinibatchKmeans algorithm
        if None perform full KMeans algorithm

    Attributes:
      centroids: torch.Tensor, shape: [n_clusters, n_features]
        cluster centroids
    '''

    def __init__(self, n_clusters, max_iter=100, tol=1e-8, init="k_means++", n_inits=10, mode="euclidean", minibatch=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.mode = mode
        self.minibatch = minibatch
        self.init = init
        self.n_inits = n_inits

        self.centroids = None
        self.inertia_ = 0

    def kmeans_plus_plus(self, x, K):
        # Kmeans++ initialization
        N, D = x.shape
        c = torch.empty(K, D, dtype=x.dtype, device=x.device)
        # 1. Choose one center uniformly at random among the data points.
        ind = int(torch.floor(torch.rand(1) * N))
        c[0, :] = x[ind, :]
        # 2. For each data point x not chosen yet, compute D(x)^2,
        #    the squared distance between x and the nearest center that has already been chosen.
        # N.B. sq_dists is initialized with infinity values and will be updated through iterations
        sq_dists = 1 / torch.zeros(N, device=x.device)
        # N.B. invarangeN below is used later in step 3
        invarangeN = torch.arange(N, 0, -1, device=x.device, dtype=torch.float32)
        for k in range(K - 1):
            sq_dists = torch.minimum(sq_dists, ((x - c[k, :]) ** 2).sum(-1))
            # 3. Choose one new data point at random as a new center,
            #    using a weighted probability distribution where a point x
            #    is chosen with probability proportional to D(x)^2.
            distrib = torch.cumsum(sq_dists, dim=0)
            ind = torch.argmax(invarangeN * (float(torch.rand(1)) * distrib[-1] < distrib))
            c[k + 1, :] = x[ind, :]
        return c

    @staticmethod
    def cos_sim(a, b):
        """
          Compute cosine similarity of 2 sets of vectors
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        a_norm = a.norm(dim=-1, keepdim=True)
        b_norm = b.norm(dim=-1, keepdim=True)
        a = a / (a_norm + 1e-8)
        b = b / (b_norm + 1e-8)
        return a @ b.transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        """
          Compute euclidean similarity of 2 sets of vectors
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        assert a.size()[1] == b.size()[1]
        assert len(a.size()) == 2 and len(b.size()) == 2
        distance_matrix = ((a[:, :, None] - b.t()[None, :, :]) ** 2).sum(1)
        return distance_matrix

    def soft_sim(self, a, b):
        """
          Compute soft similarity (or minimum distance) of each vector in a with all of the vectors in b
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        soft_sim = 1 / self.euc_sim(a, b)
        soft_sim = soft_sim / soft_sim.sum(1, keepdim=True)
        return soft_sim

    def fit(self, X, centroids=None):
        """
          Combination of fit() and predict() methods.
          This is faster than calling fit() and predict() separately.
          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]
          centroids: {torch.Tensor, None}, default: None
            if given, centroids will be initialized with given tensor
            if None, centroids will be randomly chosen from X
          Return:
          labels: torch.Tensor, shape: [n_samples]
        """
        batch_size, emb_dim = X.shape
        device = X.device.type

        centroids_list = []
        inertias_list = []
        for init in range(self.n_inits):
            if centroids is None:
                if self.init == "k_means++":
                    self.centroids = self.kmeans_plus_plus(X, self.n_clusters)
                else:
                    self.centroids = X[np.random.choice(batch_size, size=[self.n_clusters], replace=False)]
            else:
                self.centroids = centroids
            num_points_in_clusters = torch.ones(self.n_clusters, device=device)

            for i in range(self.max_iter):
                # Expectation
                soft_sim_ = self.soft_sim(X, self.centroids)
                soft_sim_ = distributed_sinkhorn(soft_sim_, epsilon=0.05)
                hard_sim = torch.nn.functional.one_hot(soft_sim_.argmax(1), num_classes=self.n_clusters).float()
                c_grad = (hard_sim.t() @ X) / hard_sim.t().sum(1, keepdim=True)

                error = (c_grad - self.centroids).pow(2).sum()
                if self.minibatch is not None:
                    lr = 1 / num_points_in_clusters[:, None] * 0.9 + 0.1
                else:
                    lr = 1
                num_points_in_clusters += hard_sim.sum(0)
                self.centroids = (1-lr)*self.centroids + lr*c_grad
                if error <= self.tol:
                    break
            soft_sim_ = self.soft_sim(X, self.centroids)
            hard_sim = torch.nn.functional.one_hot(soft_sim_.argmax(1), num_classes=self.n_clusters).float()
            self.inertia_ = (hard_sim*self.euc_sim(X, self.centroids)).sum()
            inertias_list.append(self.inertia_.cpu().numpy())
            centroids_list.append(self.centroids)

        best_init_idx = np.argmin(inertias_list)
        self.inertia_ = inertias_list[best_init_idx]
        self.centroids = centroids_list[best_init_idx]
        return self

    def predict(self, X):
        """
          Predict the closest cluster each sample in X belongs to
          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]
          Return:
          labels: torch.Tensor, shape: [n_samples]
        """
        return self.soft_sim(a=X, b=self.centroids).argmax(1)

    def predict_proba(self, X):
        """
          Predict the closest cluster each sample in X belongs to
          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]
          Return:
          labels: torch.Tensor, shape: [n_samples]
        """
        return self.soft_sim(a=X, b=self.centroids)

    def fit_predict(self, X, centroids=None):
        """
          Perform kmeans clustering
          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]
        """
        self.fit(X, centroids)
        return self.predict(X)
