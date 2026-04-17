#  HiveMindRL 🐝🤖


![](assets/intro.png)

## Abstract

This is the codebase for the experiments pertaining to "HiveMind is a single RL agent". In this paper we show that decentralised local imitation rules aggregate into a Single RL agent.  

**Talk:** 

[![Talk](https://img.youtube.com/vi/PpSpdPhTOsk/0.jpg)](https://www.youtube.com/live/PpSpdPhTOsk)

---

## Repository Structure

```
src/
├── bandit.py                  # Bandit environments (Linear, Sigmoid, Congestion)
├── reinforcement_learning.py  # RL algorithms: CL, MCL, P-CL, P-MCL
├── population_simulation.py   # Population dynamics: Imitation of Success, Weighted Voter Rule, Majority Rule
├── analytical_solutions.py    # Closed-form replicator dynamics (TRD, MRD)
├── rl.ipynb                   # RL bandit experiments
├── populations.ipynb          # Population simulation experiments
└── plots.ipynb                # Supplementary/exploratory plots
```

---

## Notebooks

### `src/rl.ipynb` — RL Bandit Experiments

Generates figures comparing the **Cross Learning (CL)** and **Maynard Cross Learning (MCL)** bandit algorithms against closed-form analytical baselines (TRD and MRD replicator dynamics) across three reward distributions (Low / Middle / High q_a's).

| Section | Description | Output |
|---|---|---|
| Streaming RL | Single-environment CL and MCL at two learning rates (α = 0.001, 0.1) over up to 150k steps | `streaming_rl_experiments.pdf` |
| Parallel RL | Multi-environment P-CL and P-MCL varying parallel environments B ∈ {10, 1000} | `parallel_rl_experiments.pdf` |
| Congestion | CL/MCL on a two-action congestion bandit; includes interactive reward histogram widget | `congestion_experiment_rl.pdf` |

---

### `src/populations.ipynb` — Population Simulation Experiments

Generates figures comparing **population simulations** (Weighted Voter Rule and Imitation of Success) against the same TRD/MRD analytical baselines.

| Section | Description | Output |
|---|---|---|
| Population size | R_wvoter and R_success at N ∈ {10, 1000} across all three reward distributions | `population_experiments.pdf` |
| Neighbourhood size | R_wvoter with varying neighbourhood size M ∈ {2, 10, 1000} at fixed N=1000 | `nei_experiments.pdf` |
| Hybrid algorithms | Deterministic/stochastic Imitation of Success combined with R_wvoter | `hybrid_experiments.pdf` |
| Congestion | R_success and R_wvoter on the two-action congestion bandit at N=2000 | `congestion_experiments_pop.pdf` |

---

### `src/plots.ipynb` — Supplementary Plots

Standalone exploratory figures probing specific algorithmic properties.

| Cell | Description | Output |
|---|---|---|
| MCL with many alphas | MCL convergence at α ∈ {0.001, 0.05, 0.01} vs. P-MCL on evenly-spaced bandit | `MCL_with_many_alphas.pdf` |
| WVR neighbourhood sweep | R_wvoter with M ∈ {1, 2, 5, 10, 500} at N=500 | `wvr_with_many_nei_sizes.pdf` |
| WVR population sweep | R_wvoter convergence for N ∈ {10, 20, 50, 100, 500} | `R_wvoter_scenario_evenly_spaced.pdf` |
| Majority rule population sweep | Majority Rule for N ∈ {10, 20, 50, 100, 1000} | `Majority_population_scenario_evenly_spaced.pdf` |
| Majority rule neighbourhood sweep | Majority Rule with M ∈ {2, 10, 50, 100, 500} | `majority_with_many_nei_sizes.pdf` |
| Majority rule vote sweep | Varies votes S ∈ {1, 3, 10, 20, 100, 10000, ∞}; S=1 recovers WVR, S→∞ recovers deterministic majority | `majority_rule.pdf` |
| Frankenstein bee | Hybrid: deterministic/stochastic Imitation of Success with R_wvoter vs. TRD/MRD | `frankstein_bee.pdf` |

---

## Algorithms

| Symbol | Name | Parameters |
|---|---|---|
| CL | Cross Learning | `steps` — simulation steps<br>`seeds` — independent runs<br>`alpha` — learning rate<br>`bandit` — environment |
| MCL | Maynard Cross Learning | `steps` — simulation steps<br>`seeds` — independent runs<br>`alpha` — learning rate<br>`alpha_baseline` — baseline tracker rate<br>`bandit` — environment |
| P-CL | Parallel Cross Learning | `steps` — simulation steps<br>`seeds` — independent runs<br>`parallel_envs` — number of parallel environments (B)<br>`bandit` — environment |
| P-MCL | Parallel Maynard Cross Learning | `steps` — simulation steps<br>`seeds` — independent runs<br>`parallel_envs` — number of parallel environments (B)<br>`bandit` — environment |
| R_success | Imitation of Success | `steps` — simulation steps<br>`population_size` — N<br>`iterations` — independent runs<br>`deterministic` — switch rule (stochastic or deterministic)<br>`stop_if_end` — halt at convergence |
| R_wvoter | Weighted Voter Rule | `steps` — simulation steps<br>`population_size` — N<br>`iterations` — independent runs<br>`neighbourhood_size` — M<br>`switch` — update mode (bee / is_det / is_stoc)<br>`stop_if_end` — halt at convergence |
| R_majority | Majority Rule | `steps` — simulation steps<br>`population_size` — N<br>`iterations` — independent runs<br>`neighbourhood_size` — M<br>`number_of_votes` — S (votes per decision; S=1 recovers WVR, S→∞ is deterministic majority)<br>`stop_if_end` — halt at convergence |
| TRD | Taylor Replicator Dynamic | `steps` — simulation steps<br>`delta` — time step<br>`bandit` — environment |
| MRD | Maynard Replicator Dynamic | `steps` — simulation steps<br>`delta` — time step<br>`bandit` — environment |

---

## Environments

| Name | Parameters |
|---|---|
| BanditLinear | `n_action` — number of arms<br>`gap` — reward noise (uniform ± gap around q_star)<br>`name` — reward preset: `near zero` (q_star ∈ [0.1, 0.4]), `evenly spaced` (q_star ∈ [0.4, 0.7]), `near one` (q_star ∈ [0.6, 0.9]) |
| BanditSigmoid | `n_action` — number of arms<br>`mean` — mean of the q_star distribution<br>`std` — standard deviation of q_star and rewards<br>`name` — same presets as BanditLinear but in logit space; rewards passed through sigmoid |
| BanditCongestion | `n_action` — number of arms (2)<br>`q_star` — base mean rewards per action<br>`congestion_factors` — per-action congestion weight ω; effective reward = q_star − ω × policy<br>`gap` — reward noise (uniform ± gap) |

---

## Future Work

- **Cross-inhibition:** Extending the equivalence to populations that use cross-inhibitory signalling, where individuals actively suppress competing options.
- **Multi-population multi-agent RL:** Generalising the framework to settings with multiple interacting populations, each acting as a macro-agent, giving rise to multi-agent RL dynamics.
- **Algorithm design — RL → imitation rules:** Systematically deriving local imitation rules from a target RL algorithm, enabling principled design of swarm behaviours.
- **Algorithm design — imitation rules → RL:** Going in the other direction: given a set of imitation rules, characterising the emergent macro-agent and its learning algorithm.
- **LOLA-like macro-agents:** Designing imitation rules whose emergent macro-agent corresponds to a Learning with Opponent-Aware Learning (LOLA) agent, connecting swarm behaviour to opponent-shaping in multi-agent RL.

---

## Citation

If you use this code, please cite:

```bibtex
@article{soma2024hivemind,
  title={The Hive Mind is a Single Reinforcement Learning Agent},
  author={Soma, Karthik and Bouteiller, Yann and Hamann, Heiko and Beltrame, Giovanni},
  journal={arXiv preprint arXiv:2410.17517},
  year={2024},
  doi={10.48550/arXiv.2410.17517}
}
```

