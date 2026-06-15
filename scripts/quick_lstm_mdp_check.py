from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.phase_B_multiseeds import CartPoleArgs, train_ppo_lstm


def main():
    args = CartPoleArgs(
        seed=1,
        total_timesteps=500_000,
        num_envs=8,
        num_steps=128,
        num_minibatches=4,
        learning_rate=2.5e-4,
        update_epochs=4,
        ent_coef=0.01,
        hidden_size=128,
        lstm_hidden_size=128,
    )

    model, history, duration = train_ppo_lstm(
        args,
        use_masked=False,
        verbose=True,
    )

    returns = history["return"]

    out_dir = ROOT / "results" / "csv" / "lstm_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "step": history["step"],
        "return": history["return"],
        "config": "ppo_lstm_mdp_stronger",
        "seed": args.seed,
    })

    out_csv = out_dir / "ppo_lstm_mdp_stronger_seed1.csv"
    df.to_csv(out_csv, index=False)

    if len(returns) >= 50:
        final_return = sum(returns[-50:]) / 50
    elif len(returns) > 0:
        final_return = sum(returns) / len(returns)
    else:
        final_return = 0.0

    print("\n" + "=" * 60)
    print("PPO-LSTM stronger sanity check sur CartPole complet")
    print("=" * 60)
    print("Nombre episodes:", len(returns))
    print("Retour moyen final:", final_return)
    print("Durée:", duration)
    print("CSV sauvegardé:", out_csv)
    print("=" * 60)


if __name__ == "__main__":
    main()