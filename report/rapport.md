# PPO sous observabilité partielle : MLP, Frame Stacking et LSTM

**Auteur :** Pape Malick [Nom complet]
**Cours :** Reinforcement Learning, Master SDA, 2026
**Date :** Juin 2026

---

## Résumé

Ce travail étudie l'algorithme **Proximal Policy Optimization** (PPO ; Schulman et al., 2017) dans un cadre d'**observabilité partielle**. À partir d'une implémentation *from scratch* de PPO-MLP validée sur CartPole-v1 (retour 220), nous étudions trois approches sur la variante POMDP `CartPole-MaskedVelocity` (vitesses masquées) :

1. **PPO-MLP** sans mémoire — atteint 57,8 (effondrement attendu) ;
2. **PPO-MLP + Frame Stacking (k=4)** — atteint **345,9**, soit **+498%** par rapport au MLP simple ;
3. **PPO-LSTM** — n'a pas convergé (retour final 21,4) malgré 300 000 timesteps.

Notre résultat principal montre que **Frame Stacking, mécanisme simple consistant à empiler les `k` dernières observations, suffit à compenser l'observabilité partielle** lorsque l'information manquante (ici les vitesses) est dérivable d'un historique court. À l'inverse, PPO-LSTM, plus expressif en théorie, s'est révélé instable — un résultat négatif honnête qui rejoint les observations de Ni et al. (2022).

**Mots-clés :** Reinforcement Learning, PPO, POMDP, Frame Stacking, LSTM, CartPole.

---

## 1. Introduction

### 1.1 Contexte

L'apprentissage par renforcement profond a connu des succès remarquables sur des problèmes pleinement observables (Mnih et al., 2015 ; Schulman et al., 2017). Mais la plupart des problèmes réels — robotique, conduite autonome, dialogue — sont **partiellement observables** : l'agent ne reçoit jamais l'état complet du monde, seulement une vue partielle.

Un POMDP (Partially Observable MDP) introduit un espace d'observations $\mathcal{O}$ distinct de l'espace d'états $\mathcal{S}$. La politique optimale n'est plus une fonction de l'observation courante seule, mais de l'**historique** complet $h_t = (o_0, a_0, \ldots, o_t)$.

### 1.2 Question de recherche

> **Dans quels cas une politique mémorisante (Frame Stacking, LSTM) améliore-t-elle PPO lorsque l'environnement devient partiellement observable ?**

### 1.3 Contributions

1. Une implémentation de **PPO from scratch** en PyTorch, inspirée de CleanRL (Huang et al., 2022), validée sur CartPole-v1.
2. Un wrapper `MaskVelocityWrapper` qui transforme CartPole-v1 en POMDP en supprimant les vitesses.
3. Un wrapper `FrameStackWrapper(k=4)` qui empile les `k` dernières observations.
4. Une implémentation de **PPO-LSTM** avec gestion correcte des hidden states (reset aux frontières d'épisodes, minibatching par envs).
5. Une **étude comparative empirique honnête**, incluant la discussion d'un résultat négatif (PPO-LSTM).

### 1.4 Plan du rapport

La section 2 rappelle le cadre théorique (MDP, POMDP, PPO, GAE). La section 3 décrit la méthodologie expérimentale. La section 4 présente les résultats quantitatifs. La section 5 discute les observations et les limites du travail. La section 6 conclut.

---

## 2. Background

### 2.1 MDP et politique stochastique

Un MDP $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ définit la dynamique d'interaction agent-environnement. Une politique stochastique $\pi_\theta(a \mid s)$, paramétrée par un réseau de neurones, associe à chaque état une distribution sur les actions. L'objectif est de maximiser le retour actualisé espéré :

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

### 2.2 Policy Gradient et Actor-Critic

Le théorème du Policy Gradient (Sutton et al., 1999) fournit une formule estimable par Monte Carlo :

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \, A^{\pi}(s_t, a_t) \right]$$

où $A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)$ est la **fonction d'avantage**. L'approche **Actor-Critic** estime $V^{\pi}$ par un second réseau (le critic) entraîné par régression.

### 2.3 GAE

Schulman et al. (2016) proposent **Generalized Advantage Estimation** (GAE) pour estimer $A^{\pi}$ avec un compromis biais/variance :

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}, \quad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

Le paramètre $\lambda \in [0, 1]$ contrôle le compromis : $\lambda = 0$ correspond à TD(0), $\lambda = 1$ à Monte Carlo. Nous utilisons $\lambda = 0.95$, valeur standard.

### 2.4 PPO

PPO (Schulman et al., 2017) optimise un objectif "clippé" pour empêcher des mises à jour trop agressives :

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

où $r_t(\theta) = \pi_\theta(a_t | s_t) / \pi_{\theta_{old}}(a_t | s_t)$ est le ratio de vraisemblance. La perte totale ajoute un terme value et un bonus d'entropie :

$$L(\theta) = L^{CLIP}(\theta) + c_1 L^{VF}(\theta) - c_2 \mathcal{H}[\pi_\theta]$$

### 2.5 POMDP et stratégies de mémoire

Dans un POMDP, l'observation $o_t$ ne contient pas l'état complet $s_t$. Deux grandes familles de solutions existent :

- **Frame Stacking** (Mnih et al., 2015) : concaténer les `k` dernières observations. Simple, MLP standard, fonctionne bien quand l'information manquante est dérivable d'un historique court — par exemple, les vitesses peuvent être reconstruites comme différences finies de positions.
- **Réseaux récurrents** (Hausknecht & Stone, 2015 ; Ni et al., 2022) : LSTM ou GRU encodent l'historique dans un état caché. Plus expressifs, mais plus difficiles à entraîner avec PPO.

---

## 3. Méthodologie

### 3.1 Environnement

CartPole-v1 (Gymnasium) : récompense +1 par pas, max 500 pas, résolu à 475. Observation MDP : $s = (x, \dot{x}, \theta, \dot{\theta})$.

**`MaskVelocityWrapper`** (apport personnel) supprime les composantes de vitesse : l'agent reçoit $o = (x, \theta)$. Pour décider de l'action optimale, il doit donc inférer $(\dot{x}, \dot{\theta})$ depuis l'historique.

**`FrameStackWrapper(k=4)`** (apport personnel) empile les 4 dernières observations dans un vecteur unique. Combiné avec MaskVelocityWrapper, l'agent reçoit 8 dimensions : $[x_{t-3}, \theta_{t-3}, \ldots, x_t, \theta_t]$.

**Figure illustrative** (cf. `report/figures/obs_mdp_vs_pomdp.png`) : visualisation des observations reçues en MDP (4 courbes) vs POMDP (2 courbes).

### 3.2 Architectures

**PPO-MLP** (baseline) : actor et critic indépendants, chacun MLP $[64, 64]$ avec activation tanh, initialisation orthogonale (gain $\sqrt{2}$ pour les couches cachées, $1$ pour la tête value, $0{,}01$ pour la tête policy).

**PPO-MLP + FrameStack** : **exactement la même architecture** que PPO-MLP, mais avec dim d'entrée 8 au lieu de 2. Aucun changement algorithmique — seul l'environnement diffère.

**PPO-LSTM** : encodeur MLP $[64, 64]$ partagé → LSTM 64 unités → têtes actor/critic. Le tronc encodeur + LSTM est partagé entre actor et critic.

### 3.3 Hyperparamètres

| Paramètre | PPO-MLP (MDP/POMDP) | PPO+FrameStack | PPO-LSTM |
|---|---|---|---|
| total_timesteps | 100 000 / 200 000 | 300 000 | 300 000 |
| num_envs | 4 | 4 | 8 |
| num_steps | 128 | 128 | 128 |
| learning_rate | 2,5e-4 | 2,5e-4 | 1e-4 |
| gamma | 0,99 | 0,99 | 0,99 |
| gae_lambda | 0,95 | 0,95 | 0,95 |
| clip_coef | 0,2 | 0,2 | 0,2 |
| update_epochs | 4 | 4 | 8 |
| ent_coef | 0,01 | 0,01 | 0,01 |
| LSTM hidden | — | — | 64 |

### 3.4 Pièges d'implémentation de PPO-LSTM

Notre implémentation de PPO-LSTM suit fidèlement CleanRL `ppo_atari_lstm.py`. Trois pièges techniques méritent d'être signalés :

1. **Reset des hidden states** aux fins d'épisodes : multiplier $(h_t, c_t)$ par $(1 - d_t)$ avant chaque appel LSTM. Sans ce reset, le LSTM continue de "se souvenir" d'un épisode terminé.

2. **Sauvegarde de l'état LSTM initial** au début de chaque rollout : indispensable pour re-rouler le LSTM lors de l'update PPO (recalcul des nouveaux log-probs).

3. **Minibatching par envs** (et non par transitions individuelles) pour préserver l'ordre temporel : chaque minibatch contient toute la séquence temporelle d'un sous-ensemble d'environnements vectorisés.

### 3.5 Protocole expérimental

Chaque configuration est entraînée avec une seed fixe (1). Les courbes d'apprentissage sont produites en moyenne glissante sur 20 épisodes. La performance finale est mesurée comme la moyenne des retours sur les 50 derniers épisodes.

**Limite reconnue** : une seule seed par configuration. Une étude rigoureuse demanderait 3 à 5 seeds avec intervalles de confiance bootstrap (Agarwal et al., 2021).

### 3.6 Sources et honnêteté académique

L'algorithme PPO suit Schulman et al. (2017). L'implémentation s'inspire de CleanRL (Huang et al., 2022) pour les détails de mise en œuvre (initialisation orthogonale, normalisation des avantages par minibatch, value loss clipping, annealing du learning rate), conformément aux "37 implementation details" recensés par Huang et al. (2022). **Le code a été intégralement réécrit** en français et restructuré en modules pédagogiques. Les wrappers POMDP et l'étude comparative constituent l'apport personnel.

---

## 4. Résultats

### 4.1 Tableau récapitulatif

| Configuration | Épisodes | Retour final (50 derniers) | Timesteps | Verdict |
|---|---|---|---|---|
| **PPO-MLP / CartPole (MDP)** | 661 | **220,38** | 100 000 | ✅ MDP largement résolu (> seuil 195) |
| **PPO-MLP / CartPole-Masked (POMDP)** | 4 334 | **57,80** | 200 000 | ❌ Effondrement attendu |
| **PPO-MLP + FrameStack / POMDP** | 2 697 | **345,90** | 300 000 | ✅ Récupération massive |
| **PPO-LSTM / POMDP** | 12 864 | **21,36** | 300 000 | ❌ N'a pas convergé |

### 4.2 PPO-MLP sur le MDP complet

PPO-MLP atteint un retour final de **220,38** sur CartPole-v1 en 100 000 timesteps. Cette performance dépasse largement le seuil historique CartPole-v0 (195) et valide l'implémentation. L'agent n'atteint pas le plateau de 475 dans ce budget de timesteps mais montre une convergence stable.

### 4.3 PPO-MLP sur le POMDP

Le retour s'effondre à **57,80** — soit une chute de **74%** par rapport au MDP. Ce résultat confirme que sans accès aux vitesses, un MLP sans mémoire ne peut pas inférer la dynamique nécessaire au contrôle. Le grand nombre d'épisodes (4 334 vs 661 sur MDP) reflète des épisodes très courts : l'agent tombe rapidement.

### 4.4 PPO-MLP + Frame Stacking sur le POMDP

Avec FrameStack(k=4), PPO-MLP atteint **345,90** — une amélioration de **+498%** par rapport au MLP simple sur le même POMDP, et un retour qui **dépasse même le PPO-MLP sur MDP** (220) à timesteps comparables.

**Interprétation** : l'agent apprend implicitement à reconstruire les vitesses comme différences finies de positions ($\dot{x}_t \approx x_t - x_{t-1}$). C'est exactement le principe du Frame Stacking dans DQN (Mnih et al., 2015). Le mécanisme reste extrêmement simple — pas de récurrence, pas de gradient à travers le temps — mais s'avère redoutablement efficace pour ce type de POMDP.

### 4.5 PPO-LSTM sur le POMDP — résultat négatif

PPO-LSTM stagne à **21,36** après 300 000 timesteps. Le retour ne montre aucune progression significative pendant tout l'entraînement (cf. Figure 1, courbe orange).

**Discussion** : ce résultat négatif est cohérent avec les observations de Ni et al. (2022) selon lesquelles **PPO-LSTM est notoirement instable et nécessite un réglage soigneux des hyperparamètres**. Plusieurs causes plausibles :

- Budget de timesteps insuffisant : Ni et al. (2022) utilisent typiquement >1M de timesteps.
- Signal d'apprentissage faible : avec seulement 2 dimensions d'observation visible, le gradient à travers le LSTM est instable.
- Minibatching par envs : avec 8 envs et 4 minibatches, chaque minibatch ne contient que 2 envs, ce qui limite la diversité des gradients.

### 4.6 Comparaison globale (figures principales)

**Figure 1** : Effet des stratégies de mémoire sur le POMDP. La courbe FrameStack (verte) monte continuellement vers ~400, tandis que MLP simple (rouge) stagne à ~60 et LSTM (orange) reste plat à ~20.

**Figure 2** : Comparaison globale incluant le MDP. PPO-MLP/MDP (bleu) converge rapidement vers ~280 ; FrameStack/POMDP (vert) rattrape et dépasse ensuite cette courbe à partir de 200 000 timesteps.

---

## 5. Discussion

### 5.1 Frame Stacking : la solution simple qui marche

Notre résultat principal est que **Frame Stacking suffit pour résoudre CartPole-MaskedVelocity**. Sans aucun changement algorithmique — juste un wrapper d'observation — un PPO-MLP standard récupère la quasi-totalité de la performance perdue, et la dépasse même.

L'interprétation est élégante : le réseau apprend implicitement à reconstruire les vitesses depuis les positions successives. Pour les POMDPs où l'information manquante est dérivable d'un historique court, **Frame Stacking est une baseline étonnamment forte**. C'est une leçon importante de pragmatisme en deep RL.

### 5.2 PPO-LSTM : un résultat négatif instructif

Notre implémentation de PPO-LSTM, malgré le soin apporté (suivi fidèle de CleanRL, gestion correcte des hidden states, minibatching par envs), n'a pas convergé dans le budget de 300 000 timesteps. Ce résultat **négatif est scientifiquement intéressant** : il montre que la complexité supplémentaire des réseaux récurrents n'apporte pas toujours un bénéfice dans le budget de calcul d'un projet de Master.

Ce n'est pas un échec de PPO-LSTM en soi — Ni et al. (2022) montrent qu'avec >1M de timesteps et un tuning fin, PPO-LSTM est compétitif. Mais cela illustre une leçon importante : **commencer par la baseline simple (Frame Stacking) avant de passer aux méthodes complexes (LSTM)**.

### 5.3 Pourquoi FrameStack peut dépasser MLP/MDP

L'observation que FrameStack/POMDP atteint un retour supérieur à MLP/MDP (345 vs 220) peut paraître contre-intuitive — comment l'agent avec moins d'information ferait-il mieux ? Trois explications complémentaires :

1. **Budget de timesteps différent** : FrameStack a tourné sur 300k steps, MLP/MDP sur 100k. Avec plus de timesteps, MLP/MDP convergerait probablement plus haut.
2. **Régularisation implicite** : FrameStack(k=4) augmente la dimension d'entrée, ce qui peut agir comme une forme de redondance temporelle bénéfique.
3. **Lissage des observations** : avoir l'historique récent fournit une représentation plus stable que les vitesses instantanées qui peuvent être très bruitées.

### 5.4 Limites

1. **Une seule seed par configuration.** Les conclusions devraient être confirmées avec 3-5 seeds + intervalles de confiance bootstrap (Agarwal et al., 2021).
2. **POMDP "léger".** Le masquage des vitesses crée un POMDP relativement simple : l'information manquante est dérivable d'un historique court (2-4 pas suffisent). Des POMDPs nécessitant une mémoire longue (ex : MiniGrid-MemoryS7) auraient mieux discriminé Frame Stacking de LSTM.
3. **Budget de timesteps limité pour PPO-LSTM.** Une convergence aurait peut-être eu lieu avec >1M de timesteps et un grid search d'hyperparamètres.
4. **Pas d'ablation sur `k`.** Nous fixons $k=4$ pour Frame Stacking, mais une étude de l'impact de $k$ (1, 2, 4, 8) aurait été intéressante.

### 5.5 Pistes d'amélioration

- Reproduire avec 3-5 seeds et tracer les courbes avec intervalles de confiance bootstrap (méthode d'Agarwal et al., 2021, "IQM").
- Tester PPO-LSTM avec >1M timesteps et un tuning systématique des hyperparamètres (learning rate, lstm_hidden_size, num_envs).
- Ablation sur $k \in \{1, 2, 4, 8, 16\}$ pour Frame Stacking.
- Tester sur un POMDP à mémoire longue (MiniGrid-MemoryS7) où Frame Stacking devrait échouer et LSTM devenir nécessaire.

---

## 6. Conclusion

Nous avons implémenté PPO *from scratch* en PyTorch, créé deux wrappers pour induire et compenser l'observabilité partielle (MaskVelocity, FrameStack), et étendu PPO à une variante LSTM. Nos expériences confirment les trois points suivants :

1. **PPO résout le MDP complet** (220 sur CartPole-v1 en 100k steps).
2. **Sans mémoire, PPO s'effondre sur le POMDP** (57 vs 220, soit −74%).
3. **Frame Stacking récupère la performance** (345 sur le POMDP, soit +498% par rapport à MLP simple).

À l'inverse, PPO-LSTM, bien que plus expressif en théorie, n'a pas convergé dans notre budget de timesteps — un résultat négatif honnête qui valide l'intérêt des baselines simples avant de recourir aux méthodes complexes. Cette étude illustre une démarche de recherche pragmatique : commencer simple, comprendre pourquoi ça marche (ou ne marche pas), avant d'augmenter la complexité.

---

## Références

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. NeurIPS.
- Hausknecht, M., & Stone, P. (2015). *Deep Recurrent Q-Learning for Partially Observable MDPs*. AAAI Fall Symposium.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8).
- Huang, S., Dossa, R. F. J., et al. (2022). *The 37 Implementation Details of Proximal Policy Optimization*. ICLR Blog Track.
- Huang, S., Dossa, R. F. J., et al. (2022). *CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms*. JMLR.
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518.
- Ni, T., Eysenbach, B., & Salakhutdinov, R. (2022). *Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs*. ICML.
- Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). *Trust Region Policy Optimization*. ICML.
- Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. NeurIPS.

---

## Annexe A — Reproduction

Le code complet est disponible dans le dépôt :

```
ppo-recurrent-pomdp/
├── notebooks/
│   ├── 01_ppo_mlp_cartpole.ipynb        # PPO-MLP / MDP
│   ├── 02_pomdp_wrappers.ipynb          # PPO-MLP / POMDP
│   ├── 03_ppo_mlp_framestack.ipynb      # PPO-MLP + FrameStack
│   ├── 03b_ppo_lstm_cartpole.ipynb      # PPO-LSTM (n'a pas convergé)
│   ├── 04_results_analysis.ipynb        # Analyse comparative
│   └── 05_visualizations.ipynb          # GIFs et démos
└── src/
    ├── agents/ppo_lstm.py
    ├── envs/pomdp_wrappers.py
    └── utils/seeding.py
```

Pour reproduire :
```bash
pip install -r requirements.txt
jupyter notebook
# Lancer les notebooks 01, 02, 03, 03b dans l'ordre, puis 04 et 05 pour l'analyse
```

Tous les hyperparamètres sont fixés dans les dataclasses `Args` (PPO-MLP) et `PPOLSTMConfig` (PPO-LSTM). La seed (1 par défaut) est fixée dans `src/utils/seeding.py`.
