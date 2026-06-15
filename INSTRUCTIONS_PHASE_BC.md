# Instructions — Phase B et C (Multi-seeds + Ablation)

## Objectif

Renforcer la **rigueur scientifique** du projet avec :
1. **Multi-seeds** : 3 seeds par config CartPole → IC bootstrap 95%
2. **Ablation k** : tester FrameStack avec k ∈ {1, 2, 4, 8}
3. **Analyse finale** : courbes propres avec intervalles de confiance

## Ce qui change pour ton rapport

Avant : *"On obtient X sur 1 seed"* → **anecdote**
Après : *"On obtient X ± σ sur 3 seeds avec IC 95% [a, b]"* → **résultat scientifique**

Cette différence transforme un projet "bon" en projet "rigoureux" — c'est immédiatement reconnaissable par un correcteur.

---

## Phase B — Multi-seeds CartPole (le plus important)

### Lancement

```bash
cd ppo-recurrent-pomdp
python scripts/phase_B_multiseeds.py
```

**Durée estimée** :
- CPU : ~50 min (3 seeds × 4 configs, LSTM est le plus lent)
- GPU (Colab T4) : ~15-20 min

### Que fait le script ?

Pour **chaque** des 4 configurations :
- PPO-MLP / MDP
- PPO-MLP / POMDP (MaskVelocity)
- PPO-MLP + FrameStack / POMDP
- PPO-LSTM / POMDP

Il entraîne avec **3 seeds différentes** (1, 2, 3) et sauvegarde dans `results/csv/multiseed/`.

### Que faire si Colab déconnecte ?

Le script est **idempotent** : tu peux le relancer, il écrasera les CSV. Mais pour éviter la perte, lance plutôt :
```bash
python scripts/phase_B_multiseeds.py --seeds 1
python scripts/phase_B_multiseeds.py --seeds 2
python scripts/phase_B_multiseeds.py --seeds 3
```

---

## Phase C — Ablation sur k (rapide)

### Lancement

```bash
python scripts/phase_C_ablation_k.py
```

**Durée estimée** : 5-10 min CPU

### Que fait le script ?

Teste FrameStack avec **k ∈ {1, 2, 4, 8}** pour répondre à la question : *"Combien de frames sont vraiment nécessaires pour reconstruire les vitesses ?"*

**Hypothèse** : k=2 devrait suffire (vitesse = différence de 2 positions).

---

## Phase D — Analyse finale

Une fois B et C terminées, lance le **notebook d'analyse** :

```
notebooks/08_final_analysis_multiseeds.ipynb
```

Il produit :
- ✅ Figure principale avec **IC bootstrap 95%** (multi-seeds)
- ✅ Tableau récapitulatif avec **IQM** (Interquartile Mean)
- ✅ Courbe d'ablation sur k
- ✅ Bullet points prêts pour le rapport

Toutes les figures sont sauvées dans `report/figures/`.

---

## Workflow recommandé

```
H+0:00  Lancer Phase B (en background, ~50 min CPU ou 15 min GPU)
H+0:05  Lancer Phase C dans un autre terminal (5-10 min)
H+0:15  Phase C terminée
H+0:50  Phase B terminée
H+0:55  Lancer NB 08 d'analyse (~2 min)
H+1:00  Mise à jour du rapport avec les nouvelles figures
```

---

## Comment lancer sur Colab (recommandé pour la vitesse)

Dans un notebook Colab :

```python
# 1. Upload le zip du projet (ou clone le repo)
!git clone https://github.com/TON_USER/ppo-recurrent-pomdp.git
%cd ppo-recurrent-pomdp

# 2. Installer
!pip install -q -r requirements.txt

# 3. Active GPU : Runtime > Change runtime type > T4

# 4. Lancer Phase B
!python scripts/phase_B_multiseeds.py

# 5. Lancer Phase C
!python scripts/phase_C_ablation_k.py

# 6. Télécharger les résultats
import shutil
shutil.make_archive("results_phaseBC", "zip", "results")
from google.colab import files
files.download("results_phaseBC.zip")
```

---

## Dépannage

| Problème | Solution |
|---|---|
| `ImportError: minigrid` | `pip install minigrid` (pas nécessaire pour Phase B/C) |
| Out of memory GPU | Réduire `num_envs` de 4 à 2 dans `CartPoleArgs` |
| Trop lent en CPU | Lancer sur Colab GPU |
| Crash au milieu | Relancer juste les seeds manquantes : `--seeds 2 3` |

---

## Ce que tu auras à la fin

**12 nouveaux CSV** dans `results/csv/multiseed/` + **4 CSV** dans `results/csv/ablation_k/` + **2 figures** dans `report/figures/`.

Avec ça, ton rapport pourra dire :

> *"Les expériences ont été reproduites sur 3 seeds indépendantes. Les courbes d'apprentissage sont présentées avec leur intervalle de confiance bootstrap à 95%, suivant la méthodologie d'Agarwal et al. (2021). Les résultats sont robustes à la variabilité des seeds."*

C'est immédiatement reconnaissable par un correcteur comme **du travail sérieux**.
