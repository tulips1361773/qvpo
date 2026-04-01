import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

algo_name = "PIM-DiffTD3"
csv_path = "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_27_23_51_51_0/csv_logs/final_comparison_26_03_27_23_51_51_0.csv"

df = pd.read_csv(csv_path)
row = df.iloc[0]

x = row["final_leakage_rate"]
y = row["final_legal_snr_db"]
yerr = row.get("final_legal_snr_db_std", None)

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(x, y, s=180, marker='*', label=algo_name)

if yerr is not None:
    ax.errorbar(x, y, yerr=yerr, fmt='none', capsize=5, elinewidth=1.5)

ax.annotate(
    algo_name,
    (x, y),
    textcoords="offset points",
    xytext=(8, 8),
    fontsize=11
)

ax.set_xlabel("Leakage Rate")
ax.set_ylabel("Legitimate Sensing SNR (dB)")
ax.set_title("Trade-off between Leakage Rate and Legitimate Sensing SNR")
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))

ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig("tradeoff_scatter.png", dpi=300, bbox_inches="tight")
