import math
import numpy as np
import multiprocessing
from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft
from sklearn.base import ClusterMixin, BaseEstimator

'''
    이 함수는 입력 배열 a에 대해 Z-정규화(z-score normalization)를 수행. 즉, 평균이 0이고 표준편차가 1이 되도록 정규화합니다.
	• axis=0: 축 기준으로 계산 (기본값은 열 기준)
	• ddof=0: 표준편차 계산 시 자유도 조정값 (기본값은 표준편차 n분의 1)
'''
def zscore(a, axis = 0, ddof = 0):
    # 입력 데이터를 NumPy 배열로 변환 (복사 없이 view 형태로 변환됨)
    a = np.asanyarray(a)

    # 평균 계산: 지정된 축을 따라 평균값 계산
    mean = a.mean(axis = axis)

    # 표준편차 계산: 지정된 축을 따라 ddof를 적용하여 표준편차 계산
    std = a.std(axis = axis, ddof = ddof)

    # 다차원 배열 처리: axis가 지정되었고, 평균의 차원이 원본보다 작다면
    if axis and mean.ndim < a.ndim:
        # 평균과 표준편차 차원을 늘려서 브로드캐스팅 가능하도록 조정
        res = ((a - np.expand_dims(mean, axis = axis)) / np.expand_dims(std, axis = axis))

    else:
        # 축 지정 없이 전체 데이터에 대해 z-score 계산
        res = (a - mean) / std

    # nan → 0으로 변환 (0으로 나눈 경우 NaN이 나올 수 있음)
    return np.nan_to_num(res, nan = 0)


'''
    이 함수는 np.roll()과 유사한 동작을 수행하지만, 데이터를 이동(roll)할 때 경계를 넘는 부분은 0으로 채움니다. 즉, 일반적인 순환(rolling) 이동이 아니라 **제로 패딩을 포함한 이동(shift)**을 수행합니다.

    주요 특징
	• shift: 이동 거리. 양수면 뒤로, 음수면 앞으로 이동
	• axis: 이동할 축 (없으면 전체를 flat하게 취급)
	• 이동 후 잘려나가는 부분은 0으로 채워짐
	• 크기를 유지함
'''
def roll_zero_pad(a, shift, axis = None):
    # 입력을 NumPy 배열로 변환
    a = np.asanyarray(a)

    # 이동할 필요가 없으면 원본 그대로 반환
    if shift == 0:
        return a

    # 축 지정이 없으면 전체 원소를 1D로 간주하여 처리
    if axis is None:
        n = a.size # 총 원소 개수
        reshape = True # 나중에 원래 모양으로 되돌리기 위함
    else:
        n = a.shape[axis] # 지정된 축 길이
        reshape = False

    # 이동량이 배열 크기보다 크면, 전부 0으로 채워진 배열 반환
    if np.abs(shift) > n:
        res = np.zeros_like(a)

    # 음수 방향으로 이동 (앞쪽으로 이동 → 뒤쪽에 0 추가)
    elif shift < 0:
        shift += n # 음수를 양수로 변환 (파이썬의 음수 인덱스 보정 방식과 유사)

        # 앞쪽 값은 뒤쪽으로 밀고, 남은 앞부분은 0으로 채움
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis)) # 앞부분 채울 0
        res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)

    # 양수 방향으로 이동 (뒤쪽으로 이동 → 앞쪽에 0 추가)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))  # 뒷부분 채울 0
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)

    # 축 지정이 없었다면 원래 모양으로 다시 reshape
    if reshape:
        return res.reshape(a.shape)
    else:
        return res

'''
    이 함수는 두 시계열 텐서 x와 y에 대해, 
    계수 정규화된 교차상관(Normalized Cross-Correlation, NCC)을 3차원 텐서 기준으로 계산하는 함수.
	• 주로 다변량(multivariate) 시계열의 클러스터링이나 패턴 매칭에 사용
	• 각 채널별 FFT 기반 교차상관을 계산하고, 전체 채널 합산 후 정규화
	• 교차상관 정규화 방식: coefficient normalization (NCCₐ)
'''
def _ncc_c_3dim(data):
    # 두 개의 시계열 텐서 x, y를 unpack
    x, y = data[0], data[1] # shape: (T, D), ex: T = Time, D = Channel num
    # (0, 1) 축 = (시간, 채널)을 따라 벡터 Norm 계산 -> 각 시계열의 에너지
    den = norm(x, axis = (0, 1)) * norm(y, axis = (0, 1)) # 분모: 전체 크기

    # 분모가 너무 작으면 0으로 나누는 것을 방지하기 위해 무한대 설정
    if den < 1e-9:
        den = np.inf

    # x의 길이 (시간축 길이)
    x_len = x.shape[0]

    # FFT 연산을 위해 가장 가까운 2의 제곱수로 패딩
    fft_size = 1 << (2*x_len-1).bit_length()

    # FFT(x) * FFT(y)^* 를 통해 교차상관 계산 (각 채널별로)
    cc = ifft(fft(x, fft_size, axis = 0) * np.conj(fft(y, fft_size, axis = 0)), axis = 0)

    # 출력 결과 재정렬: 중앙 정렬 형태 [-T+1:T]
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]), axis = 0)

    # 채널별 결과를 모두 합산 -> 실수값만 취함 -> 정규화
    return np.real(cc).sum(axis = -1) / den

'''
    이 함수는 **Shape-Based Distance (SBD)**의 핵심인 시계열 정렬을 위한 최적 시프트(shift)를 찾고, 이에 따라 시계열 y를 제로 패딩(0-padding) 방식으로 정렬하는 함수입니다.
	• 실제 SBD 거리 계산은 (1 - max(NCC))이지만,
	• 이 함수는 SBD를 기반으로 y를 x에 맞게 shift만 수행함
        → 중심 계산 단계 (e.g., centroid alignment)에서 사용됨
'''
def _sbd(x, y):
    # 1. 교차상관 벡터 (normalized cross-correlation) 계산
    ncc = _ncc_c_3dim([x, y]) # shape: (2*T - 1,)

    # 2. 최대 상관 위치 찾기 (가장 유사한 시프트 지점)
    idx = np.argmax(ncc)

    # 3. y 시계열을 해당 시프트만큼 정렬 (제로 패딩 기반으로)
    #    중앙 위치: max(len(x), len(y)) - 1
    y_shift = roll_zero_pad(y, (idx + 1) - max(len(x), len(y)))

    # 4. 정렬된 y 반환
    return y_shift

'''
    이 함수는 k-Shape 알고리즘의 중심 계산 과정에서, 클러스터 내 각 시계열을 **현재 중심(cur_center)에 정렬(align)**시키는 역할을 합니다.
	• 즉, 현재 중심과 시계열 간 최대 형태 유사성 위치에 맞게 시계열을 shift
	• 만약 중심이 0 벡터이면, 정렬하지 않고 원본을 그대로 반환
'''
def collect_shift(data):
    # data[0] : 현재 시계열 x
    # data[1] : 클러스터의 중심(cur_center)
    x, cur_center = data[0], data[1]

    # 중심이 0 벡터일 경우: 초기에 중심이 정해지지 않은 상태
    if np.all(cur_center == 0):
        return x # 정렬하지 않고 그대로 반환
    else:
        # 중심이 존재하면, x를 중심(cur_center)에 맞게 정렬
        return _sbd(cur_center, x)

'''
    이 함수는 클러스터 인덱스 j에 해당하는 시계열들을 모두 현재 중심(cur_center)에 정렬한 뒤, 
    그 시계열들로부터 새로운 중심(centroid)을 계산합니다.
    이는 k-means에서 중심을 평균으로 계산하는 것과 유사하지만, 
    형태 유사도(shape similarity)를 보존하기 위해 정렬 + Rayleigh Quotient 기반 eig 중심 추출을 수행합니다.
'''
def _extract_shape(idx, x, j, cur_center):
    _a = []

    # 1. 클러스터 j에 속한 시계열만 추출하여 현재 중심에 정렬
    for i in range(len(idx)):
        if idx[i] == j:
            _a.append(collect_shift([x[i], cur_center]))

    a = np.array(_a) # shape: (num_series_in_cluster, T, D)

    # 2. 클러스터에 속한 시계열이 없으면 랜덤 시계열을 중심으로 반환
    if len(a) == 0:
        indices = np.random.choice(x.shape[0], 1)
        return np.squeeze(x[indices].copy())

    columns = a.shape[1] # 시간 길이 T

    # 3. 각 시계열을 시간축 기준 z-score 정규화
    y = zscore(a, axis = 1, ddof = 1)

    # 4. 공분산 비슷한 행렬 계산 (1차 채널 기준)
    s = np.dot(y[:, :, 0].transpose(), y[:, :, 0]) # shape: (T, T)

    # 5. Centering matrix P = I - 1/T
    p = np.empty((columns, columns))
    p.fill(1.0 / columns)
    p = np.eye(columns) - p

    # 6. Rayleigh Quotient 최적화 (중심 벡터 = 고유 벡터)
    m = np.dot(np.dot(p, s), p) # centered similarity matrix
    _, vec = eigh(m)            # 고유값 분해
    centroid = vec[:, -1]       # 가장 큰 고유값의 고유 벡터 선택

    # 7. 중심 방향 정규화 (방향 결정)
    find_distance1 = np.sum(np.linalg.norm(a - centroid.reshape((x.shape[1], 1)), axis = (1, 2)))
    find_distance2 = np.sum(np.linalg.norm(a + centroid.reshape((x.shape[1], 1)), axis = (1, 2)))

    if find_distance1 >= find_distance2:
        centroid *= -1 # 방향 반전

    # 8. 최종 중심 벡터를 z-score 정규화하여 반환
    return zscore(centroid, ddof = 1)


'''
    이 함수는 입력 시계열 데이터 x를 k개의 클러스터로 군집화하는 k-Shape 알고리즘 전체 구현입니다.
	• x: shape (N, T, D) — N개 시계열, T 길이, D 차원(채널)
	• k: 클러스터 개수
	• centroid_init: 초기 중심값 (‘zero’ or ‘random’)
	• max_iter: 최대 반복 횟수
	• n_jobs: 병렬 처리할 프로세스 수
'''
def _kshape(x, k, centroid_init = 'zero', max_iter = 100, n_jobs = 1):
    m = x.shape[0] # 시계열 수
    idx = randint(0, k, size = m) # 초기 클러스터 할당 (무작위)

    # 중심 초기화
    if centroid_init == 'zero':
        centroids = np.zeros((k, x.shape[1], x.shape[2])) # (k, T, D)
    elif centroid_init == 'random':
        indices = np.random.choice(x.shape[0], k)
        centroids = x[indices].copy()

    distances = np.empty((m, k)) # (N, k) 거리 행렬

    for it in range(max_iter):
        old_idx = idx.copy() # 이전 클러스터 할당 백업

        # Step 1: 각 클러스터 중심 업데이트 (각 채널별로)
        for j in range(k):
            for d in range(x.shape[2]):
                # 채널별로 분리하여 중심 추출
                # input: (N, T, 1), cur_center: (T, 1)
                centroids[j, :, d] = _extract_shape(
                    idx,
                    np.expand_dims(x[:, :, d], axis = 2),
                    j,
                    np.expand_dims(centroids[j, :, d], axis = 1)
                )

        # Step 2: 거리 계산 (NCC 기반 SBD 거리 계산)
        pool = multiprocessing.Pool(n_jobs)
        args = []
        for p in range(m): # 각 시계열 x[p]
            for q in range(k): # 각 클러스터 중심 centroids[q]
                args.append([x[p, :], centroids[q, :]])

        result = pool.map(_ncc_c_3dim, args) # 병렬로 NCC 계산
        pool.close()

        # Step 3: distance matrix 채우기
        r = 0
        for p in range(m):
            for q in range(k):
                distances[p, q] = 1 - result[r].max() # SBD = 1 - max NCC
                r = r + 1
        # Step 4: 가장 가까운 중심으로 재할당
        idx = distances.argmin(1)

        # 수렴 조건: 클러스터 할당이 바뀌지 않으면 종료
        if np.array_equal(old_idx, idx):
            break

    return idx, centroids # 최종 클러스터 할당 및 중심 반환

'''
    이 함수는 시계열 데이터 x를 k개의 클러스터로 k-Shape 알고리즘을 통해 군집화한 뒤,
	• 각 클러스터의 중심 시계열 (centroid)
	• 각 클러스터에 속한 시계열 인덱스 리스트
	를 튜플 형태로 구성해 클러스터 리스트로 반환합니다.
'''
def kshape(x, k, centroid_init = 'zero', max_iter = 100):
    # 1. 내부 k-Shape 알고리즘 실행 (numpy 배열 변환 후 수행)
    idx, centroids = _kshape(np.array(x), k, centroid_init = centroid_init, max_iter = max_iter)
    clusters = [] # 최종 클러스터 리스트

    # 2. 각 클러스터에 대해
    for i, centroid in enumerate(centroids):
        series = [] # 해당 클러스터에 속한 시계열 인덱스

        # idx에서 현재 클러스터 i에 속한 시계열 수집
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
        # 3. (centroid, 시계열 인덱스 리스트) 형태로 저장
        clusters.append((centroid, series))
    # 4. 모든 클러스터 반환
    return clusters

class KShapeClusteringCPU(ClusterMixin, BaseEstimator):
    labels = None
    centroids_ = None

    def __init__(self, n_clusters, centroid_init = 'zero', max_iter = 100, n_jobs = None):
        self.n_clusters = n_clusters
        self.centroid_init = centroid_init
        self.max_iter = max_iter
        if n_jobs is None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs

    def fit(self, X, y = None):
        clusters = self._fit(X, self.n_clusters, self.centroid_init, self.max_iter, self.n_jobs)
        self.labels_ = np.zeros(X.shape[0])
        self.centroids_ = np.zeros((self.n_clusters, X.shape[1], X.shape[2]))
        for i in range(self.n_clusters):
            self.labels_[clusters[i][1]] = i
            self.centroids_[i] = clusters[i][0]
        return self

    def predict(self, X):
        labels, _ = self._predict(X, self.centroids_)
        return labels

    def _predict(self, x, centroids):
        m = x.shape[0]
        idx = randint(0, self.n_clusters, size = m)
        distances = np.empty((m, self.n_clusters))

        pool = multiprocessing.Pool(self.n_jobs)
        args = []
        for p in range(m):
            for q in range(self.n_clusters):
                args.append([x[p, :], centroids[q, :]])
        result = pool.map(_ncc_c_3dim, args)
        pool.close()
        r = 0
        for p in range(m):
            for q in range(self.n_clusters):
                distances[p, q] = 1 - result[r].max()
                r = r + 1
        idx = distances.argmin(1)
        return idx, centroids[idx]

    def _fit(self, x, k, centroid_init='zero', max_iter=100, n_jobs=1):
        idx, centroids = _kshape(np.array(x), k, centroid_init=centroid_init, max_iter=max_iter, n_jobs=n_jobs)
        clusters = []
        for i, centroid in enumerate(centroids):
            series = []
            for j, val in enumerate(idx):
                if i == val:
                    series.append(j)
            clusters.append((centroid, series))

        return clusters

