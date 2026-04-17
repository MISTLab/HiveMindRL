"""bandit environemnt for RL and Population experiments."""

import numpy as np
import torch


class BanditSigmoid:
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
        Initializes the bandit with the number of actions.
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
            start = -5
            stop = -2
            self.q_star = np.linspace(start=start, stop=stop, num=self.n_action)
        elif name == "near one":
            start = 2
            stop = 5
            self.q_star = np.linspace(start=start, stop=stop, num=self.n_action)
        elif name == "evenly spaced":
            start = -5
            stop = 5
            self.q_star = np.linspace(start=start, stop=stop, num=self.n_action)

        else:
            # Randomly generated q_star values from a normal distribution
            self.q_star = np.random.normal(
                loc=self.mean, scale=self.q_star_std, size=self.n_action
            )

        self.q_star_th = torch.tensor(
            self.q_star, device=self.device, dtype=torch.float32
        )
        self.action_std_th = torch.tensor(
            self.action_std, device=self.device, dtype=torch.float32
        )

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


class BanditLinear:
    """A class representing a bandit problem with multiple actions (arms)"""

    def __init__(
        self,
        n_action: int = 10,
        gap: float = 0.1,
        name: str = None,
        device: str = "cpu",
    ) -> None:
        """
        Initializes the bandit with the number of actions
        Args:
          n_action (int): Number of actions available in the bandit (arms)
          gap: (float): unifrom distribution of numbers generated between (action mean - gap) to (action mean + gap)
          name (str): Name of the bandit, used to create specific q_star distributions
                      (e.g., "near zero", "near one", "evenly spaced").
                      If None, a n_action number of means are randomly generated between start and stop values.
          cpu (str): can be cuda or cpu.
        """
        self.n_action = n_action
        self.gap = gap
        self.device = device

        if name == "near zero":
            start = 0.1
            stop = 0.4

        elif name == "near one":
            start = 0.6
            stop = 0.9

        elif name == "evenly spaced":
            start = 0.4
            stop = 0.7

        else:
            start = np.random.uniform(low=0, high=0.65)
            stop = start + 0.3

        self.q_star = np.linspace(start=start, stop=stop, num=self.n_action)
        self.s_q_star = self.q_star
        self.q_star_th = torch.tensor(
            self.q_star, device=self.device, dtype=torch.float32
        )
        self.gap_th = torch.tensor(self.gap, device=self.device, dtype=torch.float32)

    def return_no_actions(self) -> int:
        """
        Returns the number of actions available in the bandit
        """
        return self.n_action

    def optimal_action(self) -> int:
        """
        Returns the index of the optimal action
        However, in the 3 scenarios, there can be only one optimal action
        """
        return np.argmax(self.q_star)

    def pull(
        self, action: int | np.ndarray | torch.Tensor
    ) -> float | np.ndarray | torch.Tensor:
        """
        Pulls the action and returns the reward for a single action or a vector of actions.
        If action is an int, it returns a float; if it's an ndarray, it returns a torch.Tensor.
        """
        if isinstance(action, int):
            r = np.random.uniform(
                low=self.q_star[action] - self.gap,
                high=self.q_star[action] + self.gap,
                size=1,
            )
            assert 0 <= r <= 1, "rewards are not between 0 and 1"
            return r

        if isinstance(action, np.ndarray):
            r = np.random.uniform(
                low=self.q_star[action] - self.gap, high=self.q_star[action] + self.gap
            )
            assert r.all() <= 1 and r.all() >= 0, "r array is not between 0 and 1"
            return r

        if isinstance(action, torch.Tensor):
            r = torch.distributions.uniform.Uniform(
                low=self.q_star_th[action] - self.gap_th,
                high=self.q_star_th[action] + self.gap_th,
            ).sample()

            assert r.all() <= 1 and r.all() >= 0, "r array is not between 0 and 1"
            return r

        raise TypeError("Action must be an int | numpy ndarray | or torch tensor")

    def return_exp_optimal_reward(self) -> float:
        """
        Returns the expected optimal reward, which is the maximum of the q_star values
        """
        return np.max(self.q_star)


class BanditCongestion:
    """A class simulating congestion in population experiments, where the reward of an action decreases as more individuals choose it."""

    def __init__(
        self,
        n_action: int = 2,
        congestion_factors: list[float] = [0.5, 0.5],
        q_star: list[float] = [0.5, 0.5],
        device: str = "cpu",
        gap: float = 0.1,
    ) -> None:
        """
        Initializes the bandit with the number of actions
        Args:
          n_action (int): Number of actions available in the bandit (arms), for now we will only support 2 actions
          congestion_factors (list[float]): A list of congestion factors for each action, where each factor is <= q_star - gap.
          q_star (list[float]): A list of mean rewards for each action without congestion
          cpu (str): can be cuda or cpu.
          gap (float): unifrom distribution of numbers generated between (action mean - gap) to (action mean + gap)
        """
        self.n_action = n_action
        self.device = device
        self.congestion_factors = np.array(congestion_factors, dtype=np.float32)

        # for simplicity we will assume the same q_star for both actions, which is set to q_star

        self.q_star = np.array(q_star, dtype=np.float32)
        self.gap = gap

        #
        self.q_star_th = torch.tensor(
            self.q_star, device=self.device, dtype=torch.float32
        )

        self.gap_th = torch.tensor(self.gap, device=self.device, dtype=torch.float32)

        self.congestion_factors_th = torch.tensor(
            self.congestion_factors, device=self.device, dtype=torch.float32
        )

        # maybe add linear noise like in the linear bandit

    def pull(
        self, action: int | np.ndarray | torch.Tensor, policy
    ) -> float | np.ndarray | torch.Tensor:
        """
        Pulls the action and returns the reward for a single action or a vector of actions, taking into account the population congestion.
        If action is an int, it returns a float; if it's an ndarray, it returns a torch.Tensor.
        """
        if isinstance(action, int):
            r = np.random.uniform(
                low=self.q_star[action] - self.gap,
                high=self.q_star[action] + self.gap,
                size=1,
            )
            r_congested = r - self.congestion_factors[action] * policy[action]
            assert 0 <= r_congested <= 1, "rewards are not between 0 and 1"
            return r_congested

        if isinstance(action, np.ndarray):
            r = np.random.uniform(
                low=self.q_star[action] - self.gap,
                high=self.q_star[action] + self.gap,
            )
            r_congested = r - self.congestion_factors[action] * policy[action]
            assert (
                r_congested.all() <= 1 and r_congested.all() >= 0
            ), "r array is not between 0 and 1"
            return r_congested

        if isinstance(action, torch.Tensor):
            r = torch.distributions.uniform.Uniform(
                low=self.q_star_th[action] - self.gap_th,
                high=self.q_star_th[action] + self.gap_th,
            ).sample()
            r_congested = r - self.congestion_factors_th[action] * policy[action]
            assert (
                r_congested.all() <= 1 and r_congested.all() >= 0
            ), "r array is not between 0 and 1"
            return r_congested

        raise TypeError("Action must be an int | numpy ndarray | or torch tensor")

    def return_no_actions(self) -> int:
        """
        Returns the number of actions available in the bandit
        """
        return self.n_action

    def get_s_q_star(self, policy) -> np.ndarray:
        """
        Returns the expected q_star with congestion taken into account. only for replicator dynamic
        """
        s_q_star = self.q_star - self.congestion_factors * policy
        return s_q_star

    def optimal_action(self) -> int:
        """
        In this simple case, since the rewards without congestion are the same,
        so the one with the least congestion factor will be the optimal action
        """
        return np.argmin(self.congestion_factors)
