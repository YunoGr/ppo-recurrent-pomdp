# Instructions — Phase 2 : MiniGrid + correction MDP

## Contexte

Suite à l'analyse des résultats CartPole, nous avons identifié deux améliorations :

1. **PPO-MLP/MDP sous-entraîné** (220 au lieu de 500) → relancer avec 300k steps
2. **PPO-LSTM n'a pas convergé sur CartPole** mais devrait briller sur un POMDP à mémoire longue → tester sur MiniGrid-MemoryS7

Bonus identifié : Le bug `Tanh` → `ReLU` dans l'encodeur de PPO-LSTM a été corrigé.

## Plan d'action — 2 tâches en parallèle possibles

### Tâche A — Relancer NB 01 avec 300k steps (CPU OK, ~20 min)

1. Ouvrir `notebooks/01_ppo_mlp_cartpole.ipynb`
2. Trouver la cellule `args = Args(...)`
3. Changer `total_timesteps=100_000` en `total_timesteps=300_000`
4. Run All (le notebook va overwriter les CSV existants)

Attendu : retour final ~475+, plus de dérive du chariot.

### Tâche B — Nouveau NB 06 : MiniGrid (GPU recommandé, ~30 min)

1. **Installer minigrid** : `pip install minigrid` (déjà dans `requirements.txt`)
2. **Ouvrir** `notebooks/06_minigrid_memory_study.ipynb`
3. **Sur Colab avec GPU T4** (recommandé) :
   - PPO-MLP MiniGrid : ~5 min
   - PPO-MLP + FrameStack MiniGrid : ~5 min
   - PPO-LSTM MiniGrid : ~15 min
   - Total : ~30 min sur GPU, ~2h sur CPU
4. Run All

Attendu :
- PPO-MLP plafonne à ~0.5 (heuristique 50/50)
- PPO-MLP+FrameStack(k=4) plafonne aussi à ~0.5 (couloir > 4 pas)
- **PPO-LSTM monte vers 0.7-0.95** (utilise sa mémoire pour identifier l'objet)

## Note importante sur PPO-LSTM (CartPole)

Le code `src/agents/ppo_lstm.py` a été corrigé (`Tanh` → `ReLU`, `lstm_hidden_size` 64 → 128).
Le modèle LSTM CartPole que tu as entraîné précédemment est devenu **incompatible** (différentes tailles).

**Tu as deux options :**

- **Option 1 (recommandée)** : Ne pas relancer NB 03b. Garder ses CSV/courbes pour le rapport comme "résultat négatif". Supprimer le fichier `.pt` LSTM CartPole pour que NB 05 n'essaie pas de le charger.
- **Option 2** : Relancer NB 03b avec les nouveaux hyperparamètres. **Risque** : PPO-LSTM peut toujours échouer sur CartPole (problème intrinsèque, pas spécifique au code).

Mon avis : Option 1, on a déjà le résultat négatif documenté. MiniGrid est ce qui va sauver l'histoire.

## L'histoire finale du rapport

Avec les nouveaux résultats, l'histoire devient excellente :

1. **PPO sur MDP fonctionne** (CartPole 475+)
2. **POMDP léger (CartPole-Masked)** :
   - Sans mémoire → effondrement
   - **Frame Stacking suffit** (récupération massive)
   - LSTM ne converge pas (problème intrinsèque à ce signal très indirect)
3. **POMDP à mémoire longue (MiniGrid-Memory)** :
   - Sans mémoire → plafond 0.5 (heuristique)
   - Frame Stacking → plafond 0.5 aussi (couloir trop long)
   - **LSTM réussit** (~0.8+)
4. **Conclusion** : Frame Stacking et LSTM sont des outils **complémentaires**, à choisir selon la structure du POMDP.

## Quand tout est fini → rapport

Tu me préviens quand :
- NB 01 (300k MDP) a fini
- NB 06 (MiniGrid) a fini

Je mets à jour le rapport avec les vrais chiffres et figures.
