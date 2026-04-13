# 🟦 Push Block — Reinforcement Learning Agent

A pure-Python reinforcement learning environment that replicates the **Unity ML-Agents PushBlock** specification, complete with a live Pygame visualizer, a PPO agent (PyTorch), and a Q-Learning fallback.

---

## Demo

| Training | Watching | Manual Play |
|---|---|---|
| Agent trains live, reward chart updates in real time | Watch the trained agent play at adjustable speed | Control the agent yourself with the keyboard |

---


## 📸 Screenshots

### 🖼️ Image 1
![Image 1](images/image1.png)

### 🖼️ Image 2
![Image 2](images/image2.png)

### 🖼️ Image 3
![Image 3](images/image3.png)

---

## Features

- **Matches Unity ML-Agents PushBlock spec exactly**
  - 70-dimensional continuous observation vector (14 raycasts × 5 values each)
  - 5 discrete actions: Turn CW, Turn CCW, Forward, Backward, Wait
  - Reward: `−0.0025` per step, `+1.0` on goal
  - Benchmark target: **4.5 mean reward**
- **PPO Agent (PyTorch)** — actor-critic policy network with GAE advantage estimation, entropy bonus, and gradient clipping
- **Q-Learning fallback** — runs automatically if PyTorch is not installed, using epsilon-greedy exploration with ε-decay
- **BFS planning hints** — the observation vector includes optimal push direction and ideal agent position computed via BFS, accelerating learning
- **Solvability check** — BFS confirms the episode is solvable before training begins
- **Isometric Pygame renderer** — depth-sorted 3D-style board with raycasts, particles on goal, and a live stats panel
- **Three modes**: Train, Watch, Manual

---

## Installation

```bash
pip install pygame numpy torch
```

> If PyTorch is unavailable, the agent falls back to Q-Learning automatically — only `pygame` and `numpy` are required.

---

## Usage

```bash
python pushblock_game.py
```

From the menu:

| Key | Action |
|-----|--------|
| `T` | Train the agent |
| `W` | Watch the trained agent play |
| `M` | Play manually |
| `ESC` | Return to menu / quit |

**Manual controls:**

| Key | Action |
|-----|--------|
| `↑ / ↓` | Move forward / backward |
| `Q / E` | Rotate left / right |
| `Space` | Wait |

**Watch mode:**

| Key | Action |
|-----|--------|
| `+ / −` | Increase / decrease playback speed |

---

## Environment Specification

| Property | Value |
|----------|-------|
| Grid size | 10 × 10 |
| Observation space | 82-dim float vector (70 raycast + 12 state hints) |
| Action space | Discrete(5) |
| Max steps per episode | 400 |
| Goal zone | Entire top row (row 0) |
| Step penalty | −0.0025 |
| Goal reward | +1.0 |
| Benchmark mean reward | 4.5 |

### Observation Vector Layout

| Indices | Description |
|---------|-------------|
| 0 – 69 | 14 raycasts × 5 values: `[hit_wall, hit_goal, hit_block, distance, nothing]` |
| 70 – 71 | Relative block position (row Δ, col Δ), normalised |
| 72 | Agent facing direction (0–3 → 0.0–0.75) |
| 73 | BFS distance from block to goal (normalised) |
| 74 – 75 | Agent row/col (normalised) |
| 76 – 77 | Optimal push direction (dr, dc) from BFS |
| 78 – 79 | Agent's next BFS step toward ideal push position |
| 80 | Walkable distance to ideal push position (normalised) |
| 81 | Flag: agent is already at ideal push position |

### Float Environment Parameters

| Parameter | Default |
|-----------|---------|
| `block_scale` | 2.0 |
| `dynamic_friction` | 0.0 |
| `static_friction` | 0.0 |
| `block_drag` | 0.5 |

---

## Architecture

```
pushblock_game.py
│
├── PushBlockEnv        # Grid environment (step, reset, BFS helpers)
├── PolicyNet           # Actor-critic network (shared MLP, actor head, critic head)
├── PPOTrainer          # PPO update loop with GAE, clipped surrogate, entropy bonus
├── QAgent              # Tabular Q-Learning fallback with ε-decay
└── App                 # Pygame application — menu, training loop, rendering
```

### PPO Hyperparameters

| Param | Value |
|-------|-------|
| Hidden size | 256 |
| Learning rate | 3e-4 |
| Clip ε | 0.2 |
| Epochs per update | 3 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Value loss weight | 0.5 |
| Entropy bonus | 0.02 |
| Gradient clip | 0.5 |

---

## Project Structure

```
.
├── pushblock_game.py   # All source code (single-file)
└── README.md
```

---

## Requirements

| Package | Purpose |
|---------|---------|
| `pygame` | Visualisation and input |
| `numpy` | Observations and Q-table |
| `torch` *(optional)* | PPO agent |

---

## License

MIT

## How to Run

python pushblock_game (1).py

