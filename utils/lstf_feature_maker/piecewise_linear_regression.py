from sklearn.linear_model import LinearRegression
import numpy as np

class PiecewiseLinearRegression:
    def __init__(self,
                 series: np.ndarray,
                 start_point,
                 start_point_estimator = 1,
                 end_tail_estimator = 0.75):
        self.total_len = len(series)
        self.start_point = start_point

        self.grow_point = int(self.total_len * start_point_estimator) # 변경 에정
        self.decay_point = int(self.total_len * end_tail_estimator)

        self.x1, self.y1 = np.arange(self.grow_point).reshape(-1, 1), series[:self.grow_point]
        self.x2, self.y2 = (np.arange(self.start_point, self.decay_point).reshape(-1, 1),
                            series[self.start_point: self.decay_point])
        self.x3, self.y3 = (np.arange(self.decay_point, self.total_len).reshape(-1, 1),
                            series[self.decay_point: self.total_len])

    def auto_piecewise_slopes(self):
        '''
                Piecewise Linear Regression (Library-based)
                point: split point
        '''
        if self.total_len < 5 or len(self.x1) < 2 or len(self.x2) < 2 or len(self.x3) < 2:
            return 0.0, 0.0, 0.0

        grow_linear, decay_linear, tail_linear = LinearRegression(), LinearRegression(), LinearRegression()
        grow_linear.fit(self.x1, self.y1)
        decay_linear.fit(self.x2, self.y2)
        tail_linear.fit(self.x3, self.y3)
        return float(grow_linear.coef_[0]), float(decay_linear.coef_[0]), float(tail_linear.coef_[0])

    def manual_piecewise_slopes(self):
        '''
                Piecewise Linear Regression (Approximation-based)
                * 기울기만 평균 변화율로 계산
        '''
        grow_slope = (self.y1[-1] - self.y1[0]) / (self.x1[-1] - self.x1[0] + 1e-8)
        decay_slope = (self.y2[-1] - self.y2[0] / self.x2[-1] - self.x2[0] + 1e-8)
        tail_slope = (self.y3[-1] - self.y3[0] / self.x3[-1] - self.x3[0] + 1e-8)

        return grow_slope, decay_slope, tail_slope