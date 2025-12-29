import math
import time
import numpy as np
import logging

from typing import List
from abc import ABC, abstractmethod
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern


logger = logging.getLogger(__name__)


class NumWorkersSelector(ABC):
    @abstractmethod
    def get_num_workers(self) -> int:
        pass

    @abstractmethod
    def update(self, num_workers: int, througput: float):
        pass


class GPNumWorkersSelector(NumWorkersSelector):
    def __init__(self, candidates: List[int], beta=2.0, window_size=None):
        """
        max_workers: maximum number of threads (N)
        beta: UCB exploration parameter
        window_size: if set, use sliding window of recent points
        """
        self.beta = beta
        self.window_size = window_size

        # candidate grid in log-space
        self.candidates = np.array(sorted(candidates))
        self.Z = np.log(self.candidates).reshape(-1, 1)

        # storage
        self.X_train = []
        self.y_train = []

        # GP in log-space
        #kernel = RBF(length_scale=1e3) + WhiteKernel(noise_level=1e-5)
        kernel = Matern(length_scale=5.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)

        self.history = []

    def get_num_workers(self) -> int:
        """Select next number of workers via UCB acquisition."""

        # If we have too few training points return candidates that cover the space
        if len(self.X_train) == 0:
            return self.candidates[0]
        if len(self.X_train) == 1:
            return self.candidates[-1]
        if len(self.X_train) == 2:
            return self.candidates[len(self.candidates)//2]

        # predict posterior on all candidates
        mu, std = self.gp.predict(self.Z, return_std=True)

        self.history.append((mu, std))

        # UCB acquisition
        acq = mu + self.beta * std
        max_acq = np.max(acq)
        max_indices = np.where(acq == max_acq)[0]
        idx = np.random.choice(max_indices)
        selected_candidate = int(self.candidates[idx])

        log_level = logging.DEBUG
        if logger.isEnabledFor(log_level):
            metric_msg = []
            for i, (mu_i, std_i, acq_i) in enumerate(zip(mu, std, acq)):
                metric_msg.append(
                    f"num_workers={self.candidates[i]}: estimated_throughput={mu_i:.3f} +/- {std_i:.3f}, acq={acq_i:.3f}"
                )
            logger.log(level=log_level, msg=", ".join(metric_msg))

        return selected_candidate

    def update(self, num_workers: int, throughput: float) -> None:
        """Add observed throughput for given worker count."""
        self.X_train.append(num_workers)
        self.y_train.append(throughput)

        if self.window_size is not None:
            self.X_train = self.X_train[-self.window_size:]
            self.y_train = self.y_train[-self.window_size:]

        # We won't call predict if we have less than 3 points
        # so don't bother fitting the GP
        if len(self.X_train) < 3:
            return

        # prepare training data
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        Z_train = np.log(X).reshape(-1, 1)

        # fit GP
        self.gp.fit(Z_train, y)
