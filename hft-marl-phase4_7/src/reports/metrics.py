import numpy as np

class MetricsSuite:
    def __init__(self, eps=1e-8):
        self.eps = eps
    def sharpe(self, x):
        return (x.mean() / (x.std()+self.eps)) * np.sqrt(252)
    def sortino(self, x):
        downside = x[x<0].std() + self.eps
        return (x.mean() / downside) * np.sqrt(252)
    def max_drawdown(self, x):
        # x is per-step reward; convert to cumulative
        cum = np.cumsum(x)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        return float(dd.min())
    def cvar(self, x, q=0.95):
        if len(x)==0: return 0.0
        var = np.quantile(x, 1-q)
        tail = x[x<=var]
        return float(tail.mean()) if len(tail)>0 else float(var)
    def aggregate(self, x):
        return {
            "mean": float(x.mean()) if len(x)>0 else 0.0,
            "std": float(x.std()) if len(x)>0 else 0.0,
            "sharpe": float(self.sharpe(x)) if len(x)>1 else 0.0,
            "sortino": float(self.sortino(x)) if len(x)>1 else 0.0,
            "maxdd": float(self.max_drawdown(x)) if len(x)>1 else 0.0,
            "cvar": float(self.cvar(x)) if len(x)>1 else 0.0
        }
