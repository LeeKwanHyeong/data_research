import math
import torch
import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator
from tqdm import tqdm, trange

if torch.cuda.is_available():
    device = torch.device("cuda")
else: device = "cpu"

# Z-score normalization for tensors
'''
    Z-score normalization for tensors
    torch.mean & torch.std: Calculation mean and std along the specified axis (dim)
'''
def z_score(a, axis = 0, ddof = 0):
    mean = a.mean(dim = axis)
    std = a.std(dim = axis, unbiased = (ddof == 1)) # Standard deviation (ddof = 1 -> unbiased)

    if axis and mean.dim() < a.dim():
        x = (a - mean.unsqueeze(axis)).div(std.unsqueeze(axis))
        return x.masked_fill(torch.isnan(x), 0) # Replace NaN with 0 (from division by 0)
    else:
        x = a.sub_(mean).div(std) # In-place subtraction/division
        return x.masked_fill(torch.isnan(x), 0)

'''
    Custom roll with zero-padding
    torch.cat: concatenates tensors along a dimension
    torch.zeros: creates a tensor filled with zeros
    Shift left/right with zero padding instead of wrap-arround
'''
def roll_zero_pad(a, shift):
    if shift == 0:
        return a

    if abs(shift) > len(a):
        return torch.zeros_like(a)

    padding = torch.zeros(abs(shift), a.shape[1], device = device, dtype = torch.float32)
    if shift < 0:
        return torch.cat((a[abs(shift):], padding))
    else:
        return torch.cat((padding, a[:-shift]))

'''
    Normalized Cross-Correlation for 3D tensor (T, D)
    torch.conj: complex conjugate
    torch.ifft: inverse FFT
    torch.real: extract real part from complex numbers
    torch.norm: compute Euclidean (L2) norm
'''
def _ncc_c_3dim(x, y):
    den = torch.norm(x, p = 2, dim = (0, 1)) * torch.norm(y, p = 2, dim = (0, 1)) # norm(x) * norm(y)

    if den < 1e-9:
        den = torch.tensor(float('inf'), device = device, dtype = torch.float32)

    x_len = x.shape[0]
    fft_size = 1 << (2*x_len - 1).bit_length() # Next power of 2 for padding

    cc = torch.fft.ifft(torch.fft.fft(x, fft_size, dim = 0) * torch.conj(torch.fft.fft(y, fft_size, dim = 0)), dim = 0)
    cc = torch.cat((cc[-(x_len-1):], cc[:x_len]), dim = 0) # Re-center
    return torch.div(torch.sum(torch.real(cc), dim = -1), den) # Sum over channels and normalize

'''
    SBD: Shape-Based Distance
    Computes optimal alignment by maximum NCC and shifts y to match x
'''
def _sbd(x, y):
    ncc = _ncc_c_3dim(x, y)
    idx = ncc.argmax().item()
    y_shift = roll_zero_pad(y, (idx + 1) - max(len(x), len(y)))
    return y_shift

# Extract centroid shape for a given cluster j
'''
    Extracts clusters centroid via Rayleigh Quotient optimization
    torch.stack: stacks list of tensors into one tensor
    torch.mm: matrix multiplication
    torch.linalg.eigh: eigen decomposition for symmetric matrices
'''
def _extract_shape(idx, x, j, cur_center):
    _a = []
    for i in range(len(idx)):
        if idx[i] == j:
            if torch.sum(cur_center) == 0:
                opt_x = x[i]
            else:
                opt_x = _sbd(cur_center, x[i])
            _a.append(opt_x)
    if len(_a) == 0:
        indices = torch.randperm(x.shape[0])[:1]
        return torch.squeeze(x[indices].detach().clone())

    a =  torch.stack(_a)
    columns = a.shape[1]
    y = z_score(a, axis = 1, ddof = 1)

    s = y[:, :, 0].transpose(0, 1).mm(y[:, :, 0]) # Cross-channel similarity

    p = torch.empty((columns, columns), device = device, dtype = torch.float32)
    p.fill_(1.0 / columns)
    p = torch.eye(columns, device = device, dtype = torch.float32) - p # Centering matrix
    m = p.mm(s).mm(p)
    _, vec = torch.linalg.eigh(m, UPLO = 'U')
    centroid = vec[:, -1] # Top eigen vector

    # Resolve direction ambiguity
    find_distance_1 = torch.norm(a.sub(centroid.reshape((x.shape[1], 1))), 2, dim = (1, 2)).sum()
    find_distance_2 = torch.norm(a.add(centroid.reshape((x.shape[1], 1))), 2, dim = (1, 2)).sum()

    if find_distance_1 >= find_distance_2:
        centroid.mul_(-1)

    return z_score(centroid, ddof = 1)

'''
    Main training loog for k-Shape algorithm
    Initializes centroids and iteratively refines assignments
'''
def _kshape(x, k, centroid_init = 'zero', max_iter = 100):
    m = x.shape[0]
    idx = torch.randint(0, k, (m,), dtype = torch.float32).to(device = device)
    if centroid_init == 'zero':
        centroids = torch.zeros(k, x.shape[1], x.shape[2], device = device, dtype = torch.float32)
    elif centroid_init == 'random':
        indices = torch.randperm(x.shape[0])[:k]
        centroids = x[indices].detach().clone()
    distances = torch.empty(m, k, device = device)

    for _ in trange(max_iter, desc ='KShape Training'):
        old_idx = idx
        for j in range(k):
            for d in range(x.shape[2]):
                centroids[j, :, d] = _extract_shape(idx, torch.unsqueeze(x[:, :, d], axis=2), j, torch.unsqueeze(centroids[j, :, d], axis=1))


        for i, ts in enumerate(x):
            for c, ct in enumerate(centroids):
                dist = 1 - _ncc_c_3dim(ts, ct).max()
                distances[i, c] = dist

        idx = distances.argmin(1)
        if torch.equal(old_idx, idx):
            break

    return idx, centroids

'''
    Wraps K-shape into a readable cluster output [(centroid, members), ...]
'''
def kshape(x, k, centroid_init = 'zero', max_iter = 100):
    x = torch.tensor(x, device = device, dtype = torch.float32)
    idx, centroids = _kshape(x, k, centroid_init, max_iter)
    clusters = []

    for i, centroid in enumerate(centroids):
        series = []
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
        clusters.append((centroid, series))
    return clusters

'''
    sklearn_stype estimator wrapper for GPU-based K-shape
'''
class KShapeClusteringGPU(ClusterMixin, BaseEstimator):
    labels = None
    centroids_ = None

    def __init__(self, n_clusters, centroid_init='zero', max_iter=100):
        self.n_clusters = n_clusters
        self.centroid_init = centroid_init
        self.max_iter = max_iter

    '''
        Fit the K-shape Clustering model to data X
        Stores resulting labels_ and centroids_
    '''
    def fit(self, X, y = None):
        clusters = self._fit(X, self.n_clusters, self.centroid_init, self.max_iter)
        self.labels_ = np.zeros(X.shape[0])
        self.centroids_ = torch.zeros(self.n_clusters, X.shape[1], X.shape[2], device = device, dtype = torch.float32)
        for i in range(self.n_clusters):
            self.labels_[clusters[i][1]] = i
            self.centroids_[i] = clusters[i][0]
        return self

    '''
        Predict the cluster label for each sample in X
    '''
    def predict(self, X):
        labels, _ = self._predict(X, self.centroids_)
        return labels

    '''
        Core logic for assigning each sample in x to the closest centroid
    '''
    def _predict(self, x, centroids):
        x = torch.tensor(x, device = device, dtype = torch.float32)
        m = x.shape[0]
        k = len(centroids)
        idx = torch.randint(0, self.n_clusters, (m,), dtype = torch.float32).to(device)
        distances = torch.empty(m, self.n_clusters, device = device)

        for i, ts in enumerate(x):
            for c, ct in enumerate(centroids):
                dist = 1 - _ncc_c_3dim(ts, ct).max()
                distances[i, c] = dist

        idx = distances.argmin(1)

        return idx, centroids

    '''
        Core Clustering algorithm using torch tensors
    '''
    def _fit(self, x, k, centroid_init = 'zero', max_iter = 100):
        x = x.detach().clone().to(device=device).to(torch.float32)

        idx, centroids = _kshape(x, k, centroid_init, max_iter = max_iter)
        clusters = []
        for i, centroid in tqdm(enumerate(centroids)):
            series = []
            for j, val in enumerate(idx):
                if i == val:
                    series.append(j)
            clusters.append((centroid, series))
        return clusters

    def save(self, path: str):
        torch.save({
            'centroids': self.centroids_,
            'labels': self.labels_,
            'n_clusters': self.n_clusters,
            'max_iter': self.max_iter,
            'centroid_init': self.centroid_init,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location = device)
        self.centroids_ = data['centroids']
        self.labels = data['labels']
        self.n_clusters = data['n_clusters']
        self.max_iter = data['max_iter']
        self.centroid_init = data['centroid_init']