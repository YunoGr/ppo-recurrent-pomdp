# Setup rapide

## Option A — Local (VSCode)

```bash
cd ppo-recurrent-pomdp
python3 -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Dans VSCode : `Ctrl+Shift+P` → "Python: Select Interpreter" → `.venv/bin/python`.
Ouvre `notebooks/01_ppo_mlp_cartpole.ipynb`.

## Option B — Google Colab

```python
!git clone https://github.com/TON_USER/ppo-recurrent-pomdp.git
%cd ppo-recurrent-pomdp
!pip install -r requirements.txt
```

Puis ouvrir les notebooks. **Décommenter la cellule `pip install`** au début du NB 01.

## Ordre des notebooks

1. `01_ppo_mlp_cartpole.ipynb` (5 min CPU) — produit `results/csv/ppo_mlp_cartpole_full__seed1.csv`
2. `02_pomdp_wrappers.ipynb` (10 min CPU) — produit `results/csv/ppo_mlp_cartpole_masked__seed1.csv`
3. `03_ppo_lstm_cartpole.ipynb` (15-25 min — **GPU recommandé**) — produit `results/csv/ppo_lstm_cartpole_masked__seed1.csv`
4. `04_results_analysis.ipynb` (1 min) — agrège tout, génère les figures finales

## Vérification

```bash
python -c "import gymnasium as gym, torch; print(gym.__version__, torch.__version__, torch.cuda.is_available())"
```

## Dépannage

| Problème | Solution |
|---|---|
| ModuleNotFoundError: src | Le notebook ajoute automatiquement `ROOT` au `sys.path`. Vérifie que tu lances depuis `notebooks/` ou la racine. |
| Kernel ne voit pas les packages | Ctrl+Shift+P → "Python: Select Interpreter" → re-sélectionner le venv. |
| CartPole ne converge pas | Normal si tu coupes avant 100k. Sinon essaie seed=2 ou 3. |
| PPO-LSTM trop lent en CPU | Active GPU (Colab : Runtime → Change runtime type → T4 GPU). |
