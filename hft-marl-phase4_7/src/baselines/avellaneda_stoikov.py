"""Avellaneda–Stoikov baseline (discrete ticks). Calibrated with simple heuristics.
We set reservation price and optimal spreads, then discretise to tick offsets.
"""
import numpy as np

class AvellanedaStoikovMM:
    def __init__(self, gamma=0.1, k=1.5, sigma=0.02, tick=0.01, T=60.0):
        self.gamma = gamma  # risk aversion
        self.k = k          # order arrival parameter
        self.sigma = sigma  # mid volatility per unit time
        self.tick = tick
        self.T = T

    def quotes(self, mid, inv, t, tau=1.0):
        # reservation price with inventory term
        r = mid - inv * self.gamma * self.sigma**2 * (self.T - t)
        # optimal half-spread
        delta = self.gamma * self.sigma**2 * (self.T - t) / 2 + 1.0/self.k
        # map to tick offsets {-2..+2}
        # Suggested prices:
        bid = r - delta
        ask = r + delta
        # Convert to offsets from mid +/- spread/2 (approx)
        off_bid = int(np.clip(np.round(( (bid - (mid - self.tick)) / self.tick )), -2, 2))
        off_ask = int(np.clip(np.round(( ( (mid + self.tick) - ask) / self.tick )), -2, 2))
        return int(off_bid), int(off_ask)
