# PPO sous observabilité partielle : MLP vs LSTM

Projet de Master — Reinforcement Learning, 2026.

## Objectif

Étudier l'algorithme PPO (Schulman et al., 2017) sous **observabilité partielle**. Comparer deux approches : politique MLP vs politique récurrente LSTM, sur CartPole avec vitesses masquées.

## Question de recherche

> *Dans quels cas l'ajout d'une mémoire récurrente avec LSTM améliore-t-il PPO lorsque l'environnement devient partiellement observable ?*

## Plan d'exécution (ordre des notebooks)

| # | Notebook | Description | Temps estimé |
|---|---|---|---|
| 1 | `01_ppo_mlp_cartpole.ipynb` | PPO-MLP sur CartPole-v1 (MDP) | ~5 min CPU |
| 2 | `02_pomdp_wrappers.ipynb` | Wrapper POMDP + PPO-MLP sur CartPole-MaskedVelocity | ~10 min CPU |
| 3 | `03_ppo_lstm_cartpole.ipynb` | PPO-LSTM sur CartPole-MaskedVelocity | ~15-25 min (GPU recommandé) |
| 4 | `04_results_analysis.ipynb` | Analyse comparative + figures pour le rapport | ~1 min |
| 5 | `05_visualizations.ipynb` | **GIFs des agents en action + figures pédagogiques** | ~2 min |

## Structure

```
ppo-recurrent-pomdp/
├── notebooks/             # 4 notebooks à lancer dans l'ordre
├── src/
│   ├── agents/ppo_lstm.py   # module PPO-LSTM réutilisé par NB 03
│   ├── envs/pomdp_wrappers.py # MaskVelocityWrapper + FrameStackWrapper
│   └── utils/seeding.py     # reproductibilité
├── results/
│   ├── csv/               # courbes + CSV des runs (générés par les notebooks)
│   └── models/            # poids sauvegardés
├── report/
│   ├── rapport.md         # ← le rapport, prêt à compléter
│   └── figures/           # figures pour le rapport
├── requirements.txt
└── SETUP.md               # guide d'installation
```

## Démarrage rapide

```bash
pip install -r requirements.txt
jupyter notebook  # ou ouvrir dans VSCode
# Lancer 01 → 02 → 03 → 04 dans l'ordre
```

Voir `SETUP.md` pour les détails d'installation (VSCode, venv, Colab).

## Sources & honnêteté académique

| Élément | Source |
|---|---|
| Algorithme PPO | Schulman et al., 2017 |
| Détails d'implémentation (init orthogonale, GAE, clipping) | CleanRL (Huang et al., 2022) — code réécrit |
| Wrapper MaskVelocityWrapper | Travail personnel, inspiré de Ni et al. (2022) |
| Étude comparative MLP/LSTM sur POMDP | Travail personnel |

Le code de CleanRL n'est jamais copié-collé : il est réécrit, commenté en français et restructuré en modules.

## Références principales

- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Hausknecht, M., & Stone, P. (2015). *Deep Recurrent Q-Learning for POMDPs*. AAAI.
- Ni, T., et al. (2022). *Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs*. ICML.
- Huang, S., et al. (2022). *The 37 Implementation Details of PPO*. ICLR Blog Track.
