class ExponentialSmoothing:
    def __init__(self):
        self.alpha = 0.3

    def simple_exponential_smoothing(self, series):
        result = [series[0]] # Initial Value
        for t in range(1, len(series)):
            result.append(self.alpha * series[t] + (1 - self.alpha) * result[-1])

        return result # 마지막 값 (현재 시점까지 평활한 값)

