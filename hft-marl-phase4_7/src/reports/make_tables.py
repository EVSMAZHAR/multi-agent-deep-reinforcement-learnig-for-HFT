"""Generate CSV tables of metrics (example)."""
import os, csv
os.makedirs("reports/tables", exist_ok=True)
with open("reports/tables/summary.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["Model","Mean","Sharpe","Sortino","MaxDD","CVaR"])
    w.writerow(["MAPPO","--","--","--","--","--"])
    w.writerow(["MADDPG","--","--","--","--","--"])
    w.writerow(["A–Stoikov","--","--","--","--","--"])
print("Saved reports/tables/summary.csv")
