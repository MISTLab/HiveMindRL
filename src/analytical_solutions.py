"""replicator dynamics for analytical solutions."""

import numpy as np
from bandit import (
    BanditSigmoid,
    BanditLinear,
    BanditCongestion,
)  # Assuming Bandit class is defined in bandit.py


def replicator_dynamic(
    delta: float,
    bandit: BanditSigmoid | BanditLinear | BanditCongestion,
    steps: int,
    trd: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Taylor replicator dynamics for a bandit problem.
    Args:
        delta: Time step size for the dynamics.
        bandit: Instance of the Bandit class.
        steps: Number of steps to run the dynamics.
        trd: If True, use the Taylor replicator dynamics;
             otherwise, use the Maynard replicator dynamics.
    Returns:
        mean_fitness: Array of average fitness value over time.
        optimal_ratio: Array of the ratio of optimal action selection over time.
    """

    dt = delta
    time = np.arange(0, delta * steps, dt)
    mean_fitness = np.zeros(len(time))
    optimal_ratio = np.zeros(len(time))

    x = np.ones(bandit.n_action) * 1 / bandit.n_action


    for i in range(len(time)):
        if isinstance(bandit, BanditCongestion):
            s_qstar = bandit.get_s_q_star(x)
    
        else:
            s_qstar = bandit.s_q_star

        mean_fitness[i] = np.dot(x, s_qstar)
        optimal_ratio[i] = x[bandit.optimal_action()]
        dx = np.zeros(len(x))
        if trd:
            # Taylor replicator dynamics
            dx = x * (s_qstar - mean_fitness[i])
        else:
            # Maynard replicator dynamics
            dx = x * (s_qstar - mean_fitness[i]) / mean_fitness[i]
        x = x + dx * dt  # simulate the dynamics
    return mean_fitness, optimal_ratio


# def replicator_dynamic_non_stationary(
#     delta: float, bandit: BanditCongestion, steps: int, trd: bool
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Taylor replicator dynamics for a non-stationary bandit problem.
#     Args:
#         delta: Time step size for the dynamics.
#         bandit: Instance of the Bandit class.
#         steps: Number of steps to run the dynamics.
#         trd: If True, use the Taylor replicator dynamics;
#              otherwise, use the Maynard replicator dynamics.
#     Returns:
#         mean_fitness: Array of average fitness value over time.
#         policy_history: Array of the policy distribution over time.
#     """

#     dt = delta
#     time = np.arange(0, delta * steps, dt)
#     mean_fitness = np.zeros(len(time))
#     policy_history = np.zeros((len(time), bandit.n_action))
#     x = np.ones(bandit.n_action) * 1 / bandit.n_action

#     for i in range(len(time)):
#         policy_history[i] = x
#         s_qstar = bandit.get_s_q_star(x)
#         mean_fitness[i] = np.dot(x, s_qstar)
#         # optimal_ratio[i] = x[bandit.optimal_action()]
#         dx = np.zeros(len(x))
#         if trd:
#             # Taylor replicator dynamics
#             dx = x * (s_qstar - mean_fitness[i])
#         else:
#             # Maynard replicator dynamics
#             dx = x * (s_qstar - mean_fitness[i]) / mean_fitness[i]
#         x = x + dx * dt  # simulate the dynamics

#     return mean_fitness, policy_history
