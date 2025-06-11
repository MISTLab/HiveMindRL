"""reinforcement learning algorithms CL, MCL, P-CL, and P-MCL."""

import numpy as np
import torch
from bandit import Bandit


def streaming_cross_learning(
    steps: int, seeds: int, alpha: float, bandit: Bandit, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the streaming variant of cross-learning algorithm (CL) in a parallelized
    manner for all seeds.

    Args:
        steps (int): Number of steps for the training.
        seeds (int): Number of seeds for training.
        alpha (float): Learning rate for the policy update.
        bandit (Bandit): The bandit environment to interact with.
        device (str): cuda or cpu
    Returns:
        rewards_overall (np.ndarray): Array of rewards over time for each seed.
        optimal_action (np.ndarray): Array indicating the probability of the
                                     optimal action over time.
    """
    rewards_overall = torch.zeros((steps, seeds), device=device)
    optimal_actions = torch.zeros((steps, seeds), device=device)
    # initialize the policy with equal probabilities for each action for all seeds
    policy_all_seeds = (
        torch.ones(seeds, bandit.n_action, device=device) / bandit.n_action
    )

    optimal_action_index_th = torch.tensor(bandit.optimal_action(), device=device)

    for j, _ in enumerate(range(steps)):
        # sample actions from the policy for all the seeds for time step j
        action_all_seeds = torch.distributions.Categorical(policy_all_seeds).sample()
        # get rewards for the sampled actions for all seeds for this time step j
        rewards_all_seeds = bandit.pull(action_all_seeds)
        rewards_overall[j, :] = rewards_all_seeds
        # store the probabilites of the optimal action for all seeds at j
        optimal_actions[j, :] = policy_all_seeds[:, optimal_action_index_th]
        # update the policy for all the seeds
        mask = torch.zeros(seeds, bandit.n_action, device=device)
        mask[torch.arange(seeds), action_all_seeds] = 1
        # this is to create the cross-learning effect on the policy
        policy_all_seeds += alpha * torch.einsum(
            "ij,i->ij", mask - policy_all_seeds, rewards_all_seeds
        )

    return rewards_overall, optimal_actions


def streaming_maynard_cross_learning(
    steps: int,
    seeds: int,
    alpha: float,
    alpha_baseline: float,
    bandit: Bandit,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the streaming variant of Maynard cross-learning algorithm (MCL) in a parallelized
    manner for all the seeds.
    Args:
        steps (int): Number of steps for the training.
        seeds (int): Number of seeds for training.
        alpha (float): Learning rate for the policy update.
        alpha_basline (float): Baseline learning rate.
        bandit (Bandit): The bandit environment to interact with.
        device (str): cuda or cpu
    Returns:
        rewards_overall (np.ndarray): Array of rewards over time for each seed.
        optimal_action (np.ndarray): Array indicating the probability of the
                                     optimal action over time.
    """
    optimal_actions = torch.zeros((steps, seeds), device=device)
    rewards_overall = torch.zeros((steps, seeds), device=device)
    policy_all_seeds = torch.ones(seeds, bandit.n_action) / bandit.n_action
    optimal_action_index_th = torch.tensor(bandit.optimal_action(), device=device)
    mean_rewards = torch.zeros(seeds, device=device)

    for j, _ in enumerate(range(steps)):
        # sample actions from the policy for all the seeds for time step j
        action_all_seeds = torch.distributions.Categorical(policy_all_seeds).sample()
        # get rewards for the sampled actions for all seeds for this time step j
        rewards_all_seeds = bandit.pull(action_all_seeds)
        rewards_overall[j, :] = rewards_all_seeds
        # store the probabilites of the optimal action for all seeds at j
        optimal_actions[j, :] = policy_all_seeds[:, optimal_action_index_th]

        # this is an estimate of the value function for the current policy
        if j == 0:
            mean_rewards = rewards_all_seeds
        else:
            mean_rewards = (
                1 - alpha_baseline
            ) * mean_rewards + alpha_baseline * rewards_all_seeds

        # update the policy for all the seed
        mask = torch.zeros(seeds, bandit.n_action, device=device)
        mask[torch.arange(seeds), action_all_seeds] = 1
        policy_all_seeds += alpha * torch.einsum(
            "ij,i->ij", mask - policy_all_seeds, rewards_all_seeds / mean_rewards
        )

        # Clamp the policy to ensure probabilities are between 0 and 1
        policy_all_seeds = torch.clamp(policy_all_seeds, min=0.0, max=1.0)

    return rewards_overall, optimal_actions


def parallel_cross_learning(
    steps: int, seeds: int, bandit: Bandit, parallel_envs: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the parallel version of cross learning.
    Args:
        steps (int): Number of steps for the training.
        seeds (int): Number of seeds for training.
        bandit (Bandit): The bandit environment to interact with.
        parallel_envs (int): Number of parallel environements.
    Returns:
        rewards_overall (np.ndarray): Array of rewards over time for each seed.
        optimal_action (np.ndarray): Array indicating the probability of the
                                     optimal action over time.

    """

    optimal_actions = np.zeros((steps, seeds))
    reward_overall = np.zeros((steps, seeds))

    for i in range(seeds):
        policy = np.ones(bandit.n_action) / bandit.n_action
        for j in range(steps):
            optimal_actions[j, i] = policy[bandit.optimal_action()]
            parallel_actions = np.random.choice(
                bandit.return_no_actions(), parallel_envs, p=policy
            )

            parallel_rewards = bandit.pull(parallel_actions)
            action_mask = np.zeros((parallel_envs, bandit.n_action))
            action_mask[np.arange(parallel_envs), parallel_actions] = 1
            cases = action_mask - np.broadcast_to(
                policy, (parallel_envs, bandit.n_action)
            )
            delta_policy = np.einsum("ij,i->ij", cases, parallel_rewards.numpy())
            policy = policy + np.mean(delta_policy, axis=0)
            reward_overall[j, i] = np.mean(parallel_rewards.numpy())

    return reward_overall, optimal_actions


def parallel_maynard_cross_learning(
    steps: int, seeds: int, bandit: Bandit, parallel_envs: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the parallel version of maynard cross learning.
    Args:
        steps (int): Number of steps for the training.
        seeds (int): Number of seeds for training.
        bandit (Bandit): The bandit environment to interact with.
        parallel_envs (int): Number of parallel environements.
    Returns:
        rewards_overall (np.ndarray): Array of rewards over time for each seed.
        optimal_action (np.ndarray): Array indicating the probability of the
                                     optimal action over time.

    """

    optimal_actions = np.zeros((steps, seeds))
    reward_overall = np.zeros((steps, seeds))

    for i in range(seeds):
        policy = np.ones(bandit.n_action) / bandit.n_action
        for j in range(steps):
            optimal_actions[j, i] = policy[bandit.optimal_action()]
            parallel_actions = np.random.choice(
                bandit.return_no_actions(), parallel_envs, p=policy
            )

            parallel_rewards = bandit.pull(parallel_actions)
            action_mask = np.zeros((parallel_envs, bandit.n_action))
            action_mask[np.arange(parallel_envs), parallel_actions] = 1
            reward_mean = np.mean(parallel_rewards.numpy())

            cases = action_mask - np.broadcast_to(
                policy, (parallel_envs, bandit.n_action)
            )
            delta_policy = np.einsum(
                "ij,i->ij", cases, parallel_rewards.numpy() / reward_mean
            )
            policy = policy + np.mean(delta_policy, axis=0)
            reward_overall[j, i] = reward_mean

            # clipping of policy is still required
            policy = np.clip(policy, 0, 1)

    return reward_overall, optimal_actions
