# A Comparative Study of Transfer Learning Methods in Taxi and Ant RL Domains

This project investigates transfer learning strategies in reinforcement learning across two domains of varying complexity: the discrete Taxi domain and the continuous Ant locomotion domain. We implement and compare parameter transfer, feature transfer, policy distillation, progressive curricula, and more, using structured environments and reproducible baselines.

## 👥 Team Members

- Nadine El-Mufti — 260873899  
- Mahmoud Hegazy — 260580124  
- Aly Mohamed — 260902616

---

## 📁 Project Overview

We evaluate six key transfer learning techniques across two RL domains:

| Transfer Method       | Taxi Domain | Ant Domain |
|-----------------------|-------------|------------|
| Parameter Transfer    | ✅          | ✅         |
| Feature Transfer      | ❌          | ✅         |
| Policy Distillation   | ❌          | ✅         |
| Value Transfer        | ✅          | ❌         |
| Reward Shaping        | ✅          | ❌         |
| Progressive Transfer  | ❌          | ✅         |

- Discrete agents are trained using tabular Q-learning.
- Continuous agents are trained using PPO, SAC, or Actor-Critic.

---

## 🧪 Running Experiments

### ⚙️ 1. Install Dependencies

You’ll need Python 3.8+ and pip. We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

Make sure you have MuJoCo installed if you are running the Ant environment:

```bash
pip install mujoco gym[all] torch torchvision
```

### 🚖 2. Run Taxi Domain Experiments

Runs all transfer learning experiments on custom Taxi environments.

```bash
python taxi_experiment_full.py
```

This will:

- Train tabular Q-learning agents on source tasks.
- Apply transfer (parameter/value/reward) to target tasks.
- Log learning curves and results.

### 🐜 3. Run Ant Domain Experiments

Runs all experiments on Ant variants using neural PPO/SAC agents.

```bash
python ant_experiment_full.py
```

This will:

- Train neural agents on simplified Ant environments.
- Transfer policies or encoders to more complex variants.
- Apply distillation or curriculum learning where applicable.

---

## 📊 Output & Results

Both scripts will generate:

- Console output of training progress
- Logs of return, episode counts, and evaluation scores
- Summary statistics for jumpstart, asymptote, and transfer ratio

---

## 📝 Reproducibility & Notes

- All environments are seeded for determinism.
- Results are averaged over multiple seeds.
- Custom environments are implemented in `gym` and `mujoco` wrappers.

> No notebooks or interactive interfaces are included in this repo. All results are logged via script execution.

---

## 📂 File Structure (Core Scripts Only)

```
.
├── ant_experiment_full.py      # Run full Ant experiments
├── taxi_experiment_full.py     # Run full Taxi experiments
├── requirements.txt            # Python dependencies
```

---
