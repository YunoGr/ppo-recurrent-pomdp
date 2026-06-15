from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
csv_dir = ROOT / "results" / "csv" / "sb3_sanity"
fig_dir = ROOT / "report" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

files = sorted(csv_dir.glob("sb3_recurrentppo_pomdp_seed*.csv"))

if not files:
    raise FileNotFoundError(f"Aucun fichier trouvé dans {csv_dir}")

rows = []

for file in files:
    df = pd.read_csv(file)

    if "seed" not in df.columns or "eval_return" not in df.columns:
        raise ValueError(f"Colonnes attendues absentes dans {file}")

    seed = int(df["seed"].iloc[0])
    mean_return = df["eval_return"].mean()
    std_return = df["eval_return"].std()

    rows.append({
        "agent": "RecurrentPPO-LSTM",
        "implementation": "sb3-contrib",
        "env": "CartPole-MaskedVelocity",
        "seed": seed,
        "mean_eval_return": mean_return,
        "std_eval_return": std_return,
        "n_eval_episodes": len(df),
    })

summary = pd.DataFrame(rows).sort_values("seed")

overall_mean = summary["mean_eval_return"].mean()
overall_std = summary["mean_eval_return"].std()

print("\nRésumé par seed")
print(summary)

print("\nRésumé global")
print(f"Mean across seeds: {overall_mean:.2f}")
print(f"Std across seeds:  {overall_std:.2f}")

out_csv = csv_dir / "sb3_recurrentppo_pomdp_summary.csv"
summary.to_csv(out_csv, index=False)

# Figure simple : score moyen par seed
plt.figure(figsize=(7, 4))
plt.bar(
    summary["seed"].astype(str),
    summary["mean_eval_return"],
    yerr=summary["std_eval_return"],
    capsize=5,
)
plt.axhline(
    overall_mean,
    linestyle="--",
    linewidth=1,
    label=f"Mean across seeds = {overall_mean:.1f}",
)
plt.xlabel("Seed")
plt.ylabel("Mean evaluation return")
plt.title("RecurrentPPO-LSTM on CartPole-MaskedVelocity")
plt.legend()
plt.tight_layout()

out_fig = fig_dir / "sb3_recurrentppo_pomdp_eval_bar.png"
plt.savefig(out_fig, dpi=200)
plt.close()

print(f"\nCSV sauvegardé : {out_csv}")
print(f"Figure sauvegardée : {out_fig}")