import numpy as np
import torch

# from bandit import BanditCongestion, BanditLinear, BanditSigmoid

import os
from typing import Optional as optional
import multiprocessing as mp
from functools import partial
import time
from utils import init_logits, shuffle_agents

torch.set_num_threads(1)

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS (used by NumPy)
os.environ["MKL_NUM_THREADS"] = "1"  # MKL (used by NumPy on Intel)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr if used


def torch_indices(shape, device=None):
    return torch.stack(
        torch.meshgrid(*[torch.arange(s, device=device) for s in shape], indexing="ij")
    )


class Population:
    def __init__(self, size: int, device: str = "cpu", arms: int = 2) -> None:
        """
        Initialize the population with a given size and number of arms.
        Args:
            size (int): The number of individuals in the population.
            arms (int): The number of arms in the bandit problem.
            device (str): The device to use for computations (default is "cpu").
        """
        self.size = size
        self.arms = arms
        self.individuals = torch.arange(0, self.size)
        self.policies = (
            torch.ones((self.size, self.arms)) / self.arms
        )  # Assuming 2D policies for each individual

        # Assuming uniform imitation probabilities except for self-imitation
        self.imitation_matrix = (
            torch.ones((self.size, self.size)) - torch.eye(self.size)
        ) / (self.size - 1)

    def proportion_of_optimal_policy(self, optimal_policy_index: int) -> torch.Tensor:
        """
        Calculate the proportion of individuals in the population that have the optimal policy.
        Args:
            optimal_policy_index (int): The index of the optimal policy.
        Returns:
            torch.Tensor: A tensor containing the proportion of individuals with the optimal policy.
        """
        optimal_mask = torch.where(self.policies[:, optimal_policy_index] > 0.98, 1, 0)
        return optimal_mask.sum() / self.size

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Sample arms for each individual's policy in the population.
        Args:
            num_samples (int): The number of arms to sample per individual.
        Returns:
            torch.Tensor: A tensor containing the sampled arms for individuals.
        """

        sampled_arms = torch.multinomial(self.policies, num_samples, replacement=True)
        return sampled_arms

    def update_policy_rl(
        self,
        sampled_arms: torch.Tensor,
        rewards: torch.Tensor,
        alpha: float,
        is_cl: bool,
    ) -> None:
        """
        make an rl update.
        Args:
            sampled_arms (torch.Tensor): The arms sampled for each individual.
            rewards (torch.Tensor): The rewards received for the sampled arms.
            alpha (float): The learning rate for policy updates.
            is_cl (bool): Flag indicating whether to use cross-learning or maynard_cross_learning.
        """

        # Create a mask for the sampled arms
        sam_row_idx, sam_col_idx = torch_indices(sampled_arms.shape)
        mask = torch.zeros(self.size, sampled_arms.shape[1], self.arms)
        mask[sam_row_idx, sam_col_idx, sampled_arms] = 1
        # print(f"Mask: {mask}, Sampled Arms: {sampled_arms}, Rewards: {rewards}")
        # import pdb; pdb.set_trace()
        # print(f"Sampled Arms: {sampled_arms}")

        if is_cl:
            # Update policies using the rewards and learning rate for cross-learning
            rep_policies = np.repeat(
                self.policies[:, np.newaxis, :], sampled_arms.shape[1], axis=1
            )

            per_sample_change = torch.einsum(
                "ibj,ib-> ibj", mask - rep_policies, rewards
            )

            # print(f"mask - rep_policy: {mask - rep_policies}")

            # print(f"Per Sample Change: {per_sample_change.sum(axis=1)}")

            self.policies += alpha * per_sample_change.mean(axis=1)

        else:
            # Update policies using the rewards and learning rate
            # TODO: Implement the maynard_cross_learning update logic here
            pass
            # self.policies += alpha * (mask - self.policies) * rewards.unsqueeze(1)

    def update_policy_hybrid(
        self,
        sampled_arms: torch.Tensor,
        rewards: torch.Tensor,
        alpha: float,
        beta: float,
        # is_is: bool,
    ) -> None:
        """
        make an hybrid update with RL and imitation.
        Args:
            sampled_arms (torch.Tensor): The arms sampled for each individual.
            rewards (torch.Tensor): The rewards received for the sampled arms.
            is_is (bool): Flag indicating whether to use imitation of success or weighted voter model.
            alpha (float): The learning rate for RL updates.
            beta (float): The learning rate for imitation updates.
        """

        # Create a mask for the sampled arms
        sampled_arms_rows, sampled_arms_cols = torch_indices(sampled_arms.shape)
        mask = torch.zeros(self.size, sampled_arms.shape[1], self.arms)
        mask[sampled_arms_rows, sampled_arms_cols, sampled_arms] = 1

        # calculate RL update
        rep_policies = np.repeat(
            self.policies[:, np.newaxis, :], sampled_arms.shape[1], axis=1
        )

        Rl_change = torch.einsum("ibj,ib-> ibj", mask - rep_policies, rewards).mean(
            axis=1
        )

        # calculate imitation update
        imitating_partners = torch.multinomial(self.imitation_matrix, 1).squeeze(1)
        avg_rewards = rewards.mean(dim=1)
        partner_rewards = avg_rewards[imitating_partners]

        imitating_probabilities = torch.rand(self.size)

        imitation_change = torch.zeros_like(self.policies)
        imitation_mask = torch.where(
            imitating_probabilities < partner_rewards, True, False
        )

        # imitation gradient is the difference between the partner's policy and the individual's policy if the imitation is successful.
        imitation_change[imitation_mask] = (
            self.policies[imitating_partners[imitation_mask]]
            - self.policies[imitation_mask]
        )

        # print("imitation change:", imitation_change)

        self.policies += alpha * Rl_change + beta * imitation_change

        # print("Updated policies:", self.policies)


class ParametricPopulation:
    def __init__(
        self, size: int, device: str = "cpu", game: str = "rps", spread: float = 0.2
    ) -> None:
        """ """
        self.size = size
        self.pairs = self.size / 2
        self.spread = spread
        self.game = game
        self.device = device

        if self.game == "rps":
            self.payoff_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
            self.arms = 3
            self.payoff_matrix_th = torch.tensor(
                self.payoff_matrix, device=self.device, dtype=torch.float32
            )

        self.individuals = torch.arange(0, self.size)
        self.params = init_logits(N=self.size, A=self.arms, r=self.spread)
        # self.policies = (
        #     torch.ones((self.size, self.arms)) / self.arms
        # )  # Assuming 2D policies for each individual

        # Assuming uniform imitation probabilities except for self-imitation
        self.imitation_matrix = (
            torch.ones((self.size, self.size)) - torch.eye(self.size)
        ) / (self.size - 1)

    def update_policy_pg(
        self,
        alpha: float,
        lola : bool
    ) -> None:
        """
        make an rl update with lola or PG.
        Args:
            alpha (float): The learning rate for policy updates.
            lola (bool): Flag indicating whether to use lola or not.
        """
        shuffled_population, idx = shuffle_agents(
            self.params, self.size / 2, self.device
        )

        agent_pairs = shuffled_population.view(self.pairs, 2, -1)
        policy_pairs = torch.nn.functional.softmax(agent_pairs, dim=2)

        # create a matrix to do operations for both agents in one go
        double_policy_pairs = torch.cat(
            (
                policy_pairs,
                torch.index_select(
                    policy_pairs, 1, torch.LongTensor([1, 0]).to(self.device)
                ),
            ),
            axis=0,
        )

        P_1 = double_policy_pairs[:, 0, :]  # (n x a)
        P_2 = double_policy_pairs[:, 1, :]

        ones = torch.ones_like(P_1).to(self.device)
        I = torch.eye(self.arms).to(self.device)  # (a x a) = O(a)
        X_1 = I - torch.einsum("bi,bj->bij", ones, P_1)  # (n x a x a) = O(na²)
        Q_1 = torch.einsum("ik,bk->bi", self.payoff_matrix_th, P_2)  # (n x a) < O(na²)
        U_1 = torch.einsum("bik,bk->bi", X_1, Q_1)  # (n x a) < O(na³)

        naive_update = P_1 * U_1  # (n x a) = O(n x a)

        # print(f"naive update: {naive_update}")

        naive_update_theta_1 = naive_update[:nb_agent_pairs]
        naive_update_theta_2 = naive_update[nb_agent_pairs:]

        agent_pairs[:, 0, :] += naive_update_theta_1 * learning_rate
        agent_pairs[:, 1, :] += naive_update_theta_2 * learning_rate

        if lola:

            P_T = torch.einsum("bi,bj->bij", P_1, P_2)  # O(na²)
            X_2 = I - torch.einsum("bi,bj->bij", ones, P_2)  # O(na²)
            Q_2 = torch.einsum("ik,bk->bi", A, P_1)  # O(na²)
            U_21 = torch.einsum("bik,bk->bi", X_2, Q_2)  # O(na²)
            S_1 = P_2 * U_21  # O(na)
            M_1 = torch.einsum("ij,bkj->bik", A, X_2)  # O(na³)
            N_1 = torch.einsum("bij,bjk->bik", X_1, M_1)  # O(na³)
            O_1 = P_T * N_1  # O(na²)
            R_1 = torch.einsum("bik,bk->bi", O_1, S_1)  # O(na²)

            Q_2_T = torch.einsum("ik,bk->bi", A_T, P_1)  # O(na²)
            U_21_T = torch.einsum("bik,bk->bi", X_2, Q_2_T)  # O(na²)
            S_2 = P_2 * U_21_T  # O(na)

            M_2 = torch.einsum("ij,bkj->bik", A_T, X_2)  # O(na³)
            N_2 = torch.einsum("bij,bjk->bik", X_1, M_2)  # O(na³)
            O_2 = P_T * N_2  # O(na)
            R_2 = torch.einsum("bik,bk->bi", O_2, S_2)  # O(na²)

            mask = (idx < nb_lola_agents).unsqueeze(1).to(device)
            lola_update = (R_1 + R_2) * mask  # O(na)

            lola_update_theta_1 = lola_update[:nb_agent_pairs]
            lola_update_theta_2 = lola_update[nb_agent_pairs:]

            agent_pairs[:, 0, :] += lola_update_theta_1 * second_order_learning_rate
            agent_pairs[:, 1, :] += lola_update_theta_2 * second_order_learning_rate

        shuffled_population = agent_pairs.view(nb_agents, nb_actions)
        population[idx] = shuffled_population

        # print(f"mask - rep_policy: {mask - rep_policies}")

        # print(f"Per Sample Change: {per_sample_change.sum(axis=1)}")

        # self.policies += alpha * per_sample_change.mean(axis=1)

    def update_policy_hybrid(
        self,
        sampled_arms: torch.Tensor,
        rewards: torch.Tensor,
        alpha: float,
        beta: float,
        # is_is: bool,
    ) -> None:
        """
        make an hybrid update with RL and imitation.
        Args:
            sampled_arms (torch.Tensor): The arms sampled for each individual.
            rewards (torch.Tensor): The rewards received for the sampled arms.
            is_is (bool): Flag indicating whether to use imitation of success or weighted voter model.
            alpha (float): The learning rate for RL updates.
            beta (float): The learning rate for imitation updates.
        """

        # Create a mask for the sampled arms
        sampled_arms_rows, sampled_arms_cols = torch_indices(sampled_arms.shape)
        mask = torch.zeros(self.size, sampled_arms.shape[1], self.arms)
        mask[sampled_arms_rows, sampled_arms_cols, sampled_arms] = 1

        # calculate RL update
        rep_policies = np.repeat(
            self.policies[:, np.newaxis, :], sampled_arms.shape[1], axis=1
        )

        Rl_change = torch.einsum("ibj,ib-> ibj", mask - rep_policies, rewards).mean(
            axis=1
        )

        # calculate imitation update
        imitating_partners = torch.multinomial(self.imitation_matrix, 1).squeeze(1)
        avg_rewards = rewards.mean(dim=1)
        partner_rewards = avg_rewards[imitating_partners]

        imitating_probabilities = torch.rand(self.size)

        imitation_change = torch.zeros_like(self.policies)
        imitation_mask = torch.where(
            imitating_probabilities < partner_rewards, True, False
        )

        # imitation gradient is the difference between the partner's policy and the individual's policy if the imitation is successful.
        imitation_change[imitation_mask] = (
            self.policies[imitating_partners[imitation_mask]]
            - self.policies[imitation_mask]
        )

        # print("imitation change:", imitation_change)

        self.policies += alpha * Rl_change + beta * imitation_change

        # print("Updated policies:", self.policies)
