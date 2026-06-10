<div align="center">

# OM2P: Offline Multi-Agent Mean-Flow Policy

**AAMAS 2026**

</div>

## Overview

This repository is the official implementation of **OM2P: Offline Multi-Agent Mean-Flow Policy** (AAMAS 2026). OM2P is an offline multi-agent reinforcement learning (MARL) method built on an expressive **Mean Flow Policy**. The method extends [Flow Q-Learning (FQL)](https://arxiv.org/abs/2502.02538) to cooperative multi-agent settings, replacing the standard flow-matching actor with a **Mean Flow** formulation that models average velocity over a time interval rather than instantaneous velocity.

Each agent maintains its own decentralized policy and critic, trained from offline datasets collected in multi-agent environments. The core agent implementation is `FQLAgent_MeanField` in [`agents/fql_meanfield.py`](agents/fql_meanfield.py), which combines:

- **Mean Flow BC loss**: trains the actor to predict average velocity using the MeanFlow identity, with dual time inputs `(t_start, t_stop)`.
- **Q-learning loss**: optimizes the one-step flow policy toward high Q-value actions.
- **Optional OMAR-style regularization**: softmax Q-weighted action imitation for improved offline performance.

## Method

### Mean Flow Policy

Unlike standard FQL, which uses a single time variable `t` in flow matching, the Mean Flow actor (`ActorVectorMeanField` in [`utils/networks.py`](utils/networks.py)) takes two time endpoints `(t_start, t_stop)` and learns the **average velocity** over the interval. The training target follows the MeanFlow identity:

```
u_target = v - (t_start - t_stop) * du/dt
```

where `v = x_1 - x_0` is the instantaneous velocity and `du/dt` can be computed via autograd (`is_autograd=1`), finite differences (`is_autograd=0`), or set to zero (`is_autograd=2`).

At inference time, the policy performs **one-step action generation** by mapping noise directly to actions through the learned flow network.

### Multi-Agent Training

Training follows a **decentralized execution, centralized training** paradigm:

- One independent agent instance is created per cooperative agent.
- Each agent is updated with its own local observation–action transitions sampled from a shared offline replay buffer.
- Critic targets use the agent's own next-action samples from the flow policy.

## Installation

Requires Python 3.9+ and JAX. Main dependencies: `jax >= 0.4.26`, `flax >= 0.8.4`, `gymnasium == 0.29.1`.

```bash
pip install -r requirements.txt
```

For multi-agent MuJoCo environments, additionally install MuJoCo 2.1.0 and set `LD_LIBRARY_PATH` accordingly.

For multi-agent particle environments (`simple_spread`, `simple_tag`, `simple_world`):

```bash
cd multiagent-particle-envs && pip install -e .
```

## Supported Environments

| Environment | Description |
|---|---|
| `simple_spread` | Cooperative particle env — agents cover landmarks |
| `simple_tag` | Predator–prey particle env (adversary uses a fixed pretrained policy) |
| `simple_world` | Extended predator–prey scenario |
| `HalfCheetah-v2` | Multi-agent MuJoCo (via `multiagent_mujoco`) |
| OGBench / D4RL | Single-agent benchmarks (inherited from FQL) |

Offline datasets should be placed under `~/mydata/datasets/{env_name}/{data_type}/seed_{seed}_data/`. Supported `data_type` values include `expert`, `medium`, `random`, `medium-expert`, and `random-medium`.

## Usage

The main entry point is [`main.py`](main.py). The default agent config is [`agents/fql_meanfield.py`](agents/fql_meanfield.py).

### Mean Flow Policy on simple_spread (offline MARL)

```bash
python main.py \
  --env_name=simple_spread \
  --agent=agents/fql_meanfield.py \
  --agent.agent_name=fql_meanfield \
  --agent.training_task=0 \
  --data_type=expert \
  --seed=0 \
  --agent.lr=1e-3 \
  --offline_steps=1000000 \
  --agent.q_weight=1.0 \
  --agent.q_algo=ql \
  --agent.is_autograd=0 \
  --action_clip=1 \
  --eval_interval=10000
```

### Multi-agent MuJoCo

```bash
python main.py \
  --env_name=HalfCheetah-v2 \
  --agent=agents/fql_meanfield.py \
  --agent.agent_name=fql_meanfield \
  --data_type=medium \
  --seed=0 \
  --offline_steps=1000000 \
  --agent.q_algo=ql \
  --agent.is_autograd=0
```

### Baselines

Other agents can be selected via `--agent`:

```bash
# Standard FQL (flow matching, single time variable)
python main.py --env_name=simple_spread --agent=agents/fql.py --agent.agent_name=fql --data_type=expert

# IQL
python main.py --env_name=simple_spread --agent=agents/iql.py --agent.agent_name=iql --agent.q_algo=iql --data_type=expert

# CQL
python main.py --env_name=simple_spread --agent=agents/fql_meanfield.py --agent.agent_name=fql_meanfield --agent.q_algo=cql --data_type=expert
```

## Key Hyperparameters

| Flag | Description | Default |
|---|---|---|
| `--agent.q_weight` | Weight of the Q-learning loss in the actor objective | `1.0` |
| `--agent.q_algo` | Critic algorithm: `ql`, `iql`, or `cql` | `ql` |
| `--agent.is_autograd` | How to compute `du/dt`: `0` (finite diff), `1` (autograd), `2` (zero) | `1` |
| `--agent.time_selection` | Time sampling scheme: `uniform`, `exp`, `quadratic`, `beta`, `adaptive_beta` | `uniform` |
| `--agent.training_task` | Actor objective: `0` (BC+Q), `4` (Q only), `5` (BC only) | `-1` |
| `--agent.normalize_q_loss` | Normalize Q loss by \|Q\| scale | `False` |
| `--agent.omar_coe` | Weight of OMAR-style softmax Q imitation loss | `1.0` |
| `--data_type` | Offline dataset quality level | required for MARL envs |
| `--action_clip` | Clip actions to [-1, 1] during evaluation | `1` |
| `--is_data_aug` | Data augmentation mode: `0` (none), `1` (DOM2), `2` (diffusion synth), `3` (mixed) | `0` |

> **Tip**: For new environments, try `--agent.normalize_q_loss=True` and tune `--agent.q_weight` over `[0.01, 0.1, 1.0, 10.0]`.

## Code Structure

```
om2p/
├── main.py                  # Training loop (multi-agent aware)
├── agents/
│   ├── fql_meanfield.py     # Mean Flow Policy agent (main)
│   ├── fql.py               # Standard FQL agent
│   ├── fql_sep.py           # Separated FQL variant
│   ├── iql.py / ifql.py     # Baselines
│   └── rebrac.py / sac.py   # Additional baselines
├── envs/
│   └── env_utils.py         # Environment and dataset loading
├── utils/
│   ├── networks.py          # ActorVectorMeanField, Value networks
│   └── buffer.py            # Multi-agent replay buffer
├── multiagent-particle-envs/  # MPE environments
└── multiagent_mujoco/         # Multi-agent MuJoCo
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026om2p,
  title     = {OM2P: Offline Multi-Agent Mean-Flow Policy},
  author    = {Zhuoran Li and Xun Wang and Hai Zhong and Qingxin Xia and Lihua Zhang and Longbo Huang},
  booktitle = {Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year      = {2026},
}
```

## Acknowledgments

This codebase is built upon the following works:

- **[Flow Q-Learning (FQL)](https://arxiv.org/abs/2502.02538)** — Seohong Park et al. We extend FQL's flow-matching policy and Q-learning framework to offline multi-agent settings. Official code: [seohong.me/projects/fql](https://seohong.me/projects/fql/).

- **[Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)** — Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He (NeurIPS 2025). Our Mean Flow Policy adopts the MeanFlow identity and average-velocity formulation for one-step action generation in offline RL.

We also thank [OGBench](https://github.com/seohongpark/ogbench) for environment and dataset utilities inherited from the original FQL codebase.
