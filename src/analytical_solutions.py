"""replicator dynamics for analytical solutions."""

import numpy as np
from bandit import (
    BanditSigmoid,
    BanditLinear,
)  # Assuming Bandit class is defined in bandit.py


def replicator_dynamic(
    delta: float, bandit: BanditSigmoid | BanditLinear, steps: int, trd: bool
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
    """

    dt = delta
    time = np.arange(0, delta * steps, dt)
    mean_fitness = np.zeros(len(time))
    optimal_ratio = np.zeros(len(time))
    s_qstar = bandit.s_q_star
    x = np.ones(bandit.n_action) * 1 / bandit.n_action

    for i in range(len(time)):
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
