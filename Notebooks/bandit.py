"""bandit environemnt for RL and Population experiments."""

import numpy as np
import torch


class Bandit:
    """A class representing a bandit problem with multiple actions (arms)"""

    def __init__(
        self,
        n_action: int = 10,
        mean: float = 0.0,
        std: float = 1.0,
        name: str = None,
        device: str = "cpu",
    ) -> None:
        """
        Initializes the bandit with the number of actions, mean and variance for the rewards.
        Args:
          n_action (int): Number of actions available in the bandit (arms)
          mean (float): Mean for the hidden q_star distribution
          std (float): standard deviation of the rewards and q_star
          name (str): Name of the bandit, used to create specific q_star distributions
                      (e.g., "near zero", "near one", "evenly spaced").
                      If None, a random distribution is used.
          cpu (str): can be cuda or cpu.
        """
        self.n_action = n_action
        self.mean = mean
        self.q_star_std = std
        self.action_std = std
        self.num_samples = int(1e7)
        self.device = device

        if name == "near zero":
            self.q_star = np.arange(start=-5, stop=-2, step=3 / self.n_action)
        elif name == "near one":
            self.q_star = np.arange(start=1, stop=5, step=4 / self.n_action)
        elif name == "evenly spaced":
            self.q_star = np.arange(start=-3, stop=3, step=6 / self.n_action)
        else:
            # Randomly generated q_star values from a normal distribution
            self.q_star = np.random.normal(
                loc=self.mean, scale=self.q_star_std, size=self.n_action
            )

        self.q_star_th = torch.tensor(self.q_star, device=self.device)
        self.action_std_th = torch.tensor(self.action_std, device=self.device)

        self.s_q_star = self.compute_sigmoid_q_star()

    def compute_sigmoid_q_star(self) -> np.ndarray:
        """
        Computes the expected value after the sigmoid transformation for each action's q_star
        """
        s_qstar = np.zeros(self.n_action)
        for i, _ in enumerate(self.q_star):
            real_rewards = np.random.normal(
                loc=self.q_star[i], scale=self.action_std, size=self.num_samples
            )
            sigmoid_rewards = 1 / (1 + np.exp(-real_rewards))
            s_qstar[i] = np.mean(sigmoid_rewards)
        return s_qstar

    def return_no_actions(self) -> int:
        """
        Returns the number of actions available in the bandit
        """
        return self.n_action

    def optimal_action(self) -> int:
        """
        Returns the index of the optimal action
        will be the smallest index if there are multiple optimal actions
        However, in the 3 scenarios, there can be only one optimal action
        """
        return np.argmax(self.q_star)

    def pull(self, action: int | np.ndarray | torch.Tensor) -> float | torch.Tensor:
        """
        Pulls the action and returns the reward for a single action or a vector of actions.
        If action is an int, it returns a float; if it's an ndarray, it returns a torch.Tensor.
        """
        if isinstance(action, int):
            r = np.random.normal(loc=self.q_star[action], scale=self.action_std, size=1)
            r_sig = 1 / (1 + np.exp(-r))
            assert 0 <= r_sig[0] <= 1, "Sigmoid reward must be between 0 and 1"
            return r_sig[0]

        if isinstance(action, np.ndarray):
            action_th = torch.tensor(self.q_star[action])
            std_th = torch.tensor(self.action_std)
            std_th = std_th.repeat(action_th.size())
            r = torch.normal(mean=action_th, std=std_th)
            r_sig = 1 / (1 + torch.exp(-r))
            assert (
                r_sig.all() <= 1 and r_sig.all() >= 0
            ), "Sigmoid reward must be between 0 and 1"

            return r_sig

        if isinstance(action, torch.Tensor):
            action_th = self.q_star_th[action]
            std_th = self.action_std_th.repeat(action_th.size())
            r = torch.normal(mean=action_th, std=std_th)
            r_sig = 1 / (1 + torch.exp(-r))
            assert (
                r_sig.all() <= 1 and r_sig.all() >= 0
            ), "Sigmoid reward must be between 0 and 1"

            return r_sig

        raise TypeError("Action must be an int | numpy ndarray | or torch tensor")

    def return_exp_optimal_reward(self) -> float:
        """
        Returns the expected optimal reward, which is the maximum of the q_star values
        """
        return np.max(self.q_star)
