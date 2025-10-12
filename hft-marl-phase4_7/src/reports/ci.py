import numpy as np
def bootstrap_ci(x, fn=np.mean, alpha=0.05, iters=1000, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    x = np.asarray(x)
    if len(x)==0:
        return 0.0, 0.0
    stats = []
    n = len(x)
    for _ in range(iters):
        samp = x[rng.integers(low=0, high=n, size=n)]
        stats.append(fn(samp))
    stats = np.sort(stats)
    lo = stats[int((alpha/2)*iters)]
    hi = stats[int((1-alpha/2)*iters)]
    return float(lo), float(hi)
