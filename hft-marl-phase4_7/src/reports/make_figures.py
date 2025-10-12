"""Generate simple figures from saved eval outputs (example).
This script produces placeholder charts using matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np
# Example: risk-return frontier points
np.random.seed(0)
x = np.random.randn(10).cumsum()/10
y = np.random.rand(10)
plt.figure()
plt.scatter(x, y)
plt.xlabel("Expected Return")
plt.ylabel("Risk (Std)")
plt.title("Risk-Return Frontier (Example)")
plt.savefig("reports/figures/risk_return.png", dpi=150, bbox_inches='tight')
print("Saved reports/figures/risk_return.png")
