"""
Phase C - Ablation sur k pour FrameStack.

Teste FrameStack avec k in {1, 2, 4, 8} sur CartPole-MaskedVelocity.
Repond a : "Combien de frames sont reellement necessaires pour reconstruire les vitesses ?"

Hypothese : k=2 suffit (vitesse = difference de 2 positions).

Usage :
    python scripts/phase_C_ablation_k.py

Duree estimee :
    - CPU : ~5-7 min (4 valeurs de k x 1 seed x ~1-2 min/run)
    - Plus rapide que Phase B car ablations = 1 seed suffit
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reutilise les fonctions de Phase B
sys.path.insert(0, str(ROOT / "scripts"))
from phase_B_multiseeds import train_ppo_mlp, CartPoleArgs

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default=None)
    args_cli = parser.parse_args()

    OUT = Path(args_cli.out_dir) if args_cli.out_dir else ROOT / "results" / "csv" / "ablation_k"
    OUT.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(f"PHASE C - Ablation sur k de FrameStack")
    print(f"Valeurs de k : {args_cli.ks}")
    print(f"Seed : {args_cli.seed}, total_timesteps : {args_cli.total_timesteps}")
    print("="*70)

    total_start = time.time()
    for k in args_cli.ks:
        print(f"\n[k={k}] training...")
        args = CartPoleArgs(seed=args_cli.seed, total_timesteps=args_cli.total_timesteps)
        t0 = time.time()
        _, history, dur = train_ppo_mlp(args, use_masked=True, use_framestack=True, k=k)
        import numpy as np
        r50 = np.mean(history["return"][-50:]) if history["return"] else 0
        print(f"[k={k}] DONE in {dur:.0f}s | return(50 derniers)={r50:.1f}")

        df = pd.DataFrame({
            "step": history["step"],
            "return": history["return"],
            "k": k,
            "config": f"framestack_k{k}",
            "seed": args_cli.seed,
        })
        out_csv = OUT / f"framestack_k{k}__seed{args_cli.seed}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  -> {out_csv.name}")

    total_dur = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"PHASE C TERMINEE en {total_dur/60:.1f} min")
    print(f"CSVs sauvegardes dans : {OUT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
