import numpy as np
from lifelines import WeibullFitter

class WeibullFeatureMaker:
    '''
        Weibll Parameter Estimation (Library based)
    '''
    def auto_weibull_params(self, values: np.ndarray):
        if len(values) < 5:
            return 0.0, 0.0

        # 각 기간별 수요량 failure time으로 확장
        failure_times = np.repeat(np.arange(1, len(values) + 1), values.astype(int))
        if len(failure_times) < 5:
            return 0.0, 0.0

        wf = WeibullFitter()
        wf.fit(failure_times, event_observed = [1] * len(failure_times))
        return float(wf.rho_), float(wf.lambda_) # (shape = k, scale =λ)

    def manual_weibull_params(self, values: np.ndarray):
        '''
        Weibull Parameter Estimation (Approximate based)
        '''
        n = len(values)
        if n < 5:
            return 0.0, 0.0

        log_fit = np.log(np.arange(1, n+1))
        # CV based
        k = 1.086 / (np.std(log_fit) / np.mean(log_fit))
        lam = np.exp(np.mean(log_fit))
        return k, lam