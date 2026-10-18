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

        # with rewards in [0, 1], alpha + beta <= 1 keeps every policy on the simplex
        assert alpha + beta <= 1, "alpha + beta must be <= 1 to keep policies valid"

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
        assert size % 2 == 0, "population size must be even to form pairs"
        self.size = size
        self.pairs = self.size // 2
        self.spread = spread
        self.game = game
        self.device = device

        if self.game == "rps":
            self.payoff_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
            self.arms = 3
            self.payoff_matrix_th = torch.tensor(
                self.payoff_matrix, device=self.device, dtype=torch.float32
            )

        elif self.game == "hd":
            # Hawk-Dove from the Fast LOLA paper ("hd-2"), actions (Hawk, Dove); mixed Nash at p(Hawk) = 1/3
            self.payoff_matrix = np.array([[-2.0, 2.0], [0.0, 1.0]])
            self.arms = 2
            self.payoff_matrix_th = torch.tensor(
                self.payoff_matrix, device=self.device, dtype=torch.float32
            )

        elif self.game == "sh":
            # hard Stag Hunt from the Fast LOLA paper ("sh1.8"), actions (Stag, Hare); pure Nash at all-Stag and all-Hare
            self.payoff_matrix = np.array([[1.8, 0.0], [1.0, 1.0]])
            self.arms = 2
            self.payoff_matrix_th = torch.tensor(
                self.payoff_matrix, device=self.device, dtype=torch.float32
            )

        else:
            raise ValueError(f"unknown game: {self.game}")

        self.individuals = torch.arange(0, self.size)
        self.params = init_logits(N=self.size, A=self.arms, r=self.spread)
        # self.policies = (
        #     torch.ones((self.size, self.arms)) / self.arms
        # )  # Assuming 2D policies for each individual

        # Assuming uniform imitation probabilities except for self-imitation
        self.imitation_matrix = (
            torch.ones((self.size, self.size)) - torch.eye(self.size)
        ) / (self.size - 1)

    def update_param_pg(
        self,
        alpha: float,
        lola_alpha: float,
        lola : bool,
        return_update: bool
    ) -> None:
        """
        make an rl update with lola or PG.
        Args:
            alpha (float): The learning rate for policy updates.
            lola (bool): Flag indicating whether to use lola or not.
        """
        shuffled_population, idx = shuffle_agents(
            self.params, self.pairs, self.device
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

        # for imitation puposes
        policy_values =  torch.einsum("bi,bi->b", P_1, Q_1)
        # print(f"naive update: {naive_update}")

        naive_update_theta_1 = naive_update[:self.pairs]
        naive_update_theta_2 = naive_update[self.pairs:]

        agent_pairs[:, 0, :] += naive_update_theta_1 * alpha
        agent_pairs[:, 1, :] += naive_update_theta_2 * alpha

        if lola:

            P_T = torch.einsum("bi,bj->bij", P_1, P_2)  # O(na²)
            X_2 = I - torch.einsum("bi,bj->bij", ones, P_2)  # O(na²)
            Q_2 = torch.einsum("ik,bk->bi", self.payoff_matrix_th, P_1)  # O(na²)
            U_21 = torch.einsum("bik,bk->bi", X_2, Q_2)  # O(na²)
            S_1 = P_2 * U_21  # O(na)
            M_1 = torch.einsum("ij,bkj->bik", self.payoff_matrix_th, X_2)  # O(na³)
            N_1 = torch.einsum("bij,bjk->bik", X_1, M_1)  # O(na³)
            O_1 = P_T * N_1  # O(na²)
            R_1 = torch.einsum("bik,bk->bi", O_1, S_1)  # O(na²)

            Q_2_T = torch.einsum("ik,bk->bi", self.payoff_matrix_th.T, P_1)  # O(na²)
            U_21_T = torch.einsum("bik,bk->bi", X_2, Q_2_T)  # O(na²)
            S_2 = P_2 * U_21_T  # O(na)

            M_2 = torch.einsum("ij,bkj->bik", self.payoff_matrix_th.T, X_2)  # O(na³)
            N_2 = torch.einsum("bij,bjk->bik", X_1, M_2)  # O(na³)
            O_2 = P_T * N_2  # O(na)
            R_2 = torch.einsum("bik,bk->bi", O_2, S_2)  # O(na²)

            # mask = (idx < nb_lola_agents).unsqueeze(1).to(self.device)
            lola_update = (R_1 + R_2) 

            lola_update_theta_1 = lola_update[:self.pairs]
            lola_update_theta_2 = lola_update[self.pairs:]

            agent_pairs[:, 0, :] += lola_update_theta_1 * lola_alpha * alpha
            agent_pairs[:, 1, :] += lola_update_theta_2 * lola_alpha * alpha

        shuffled_population = agent_pairs.view(self.size, self.arms)

        if return_update:
            # want to share these values with the shuffling
            # shuffled_population row k belongs to agent idx[k]; policy_values and P_1 are stacked
            # (slot-0 agents, then slot-1 agents), so their rows belong to idx[0::2] then idx[1::2]
            stacked_idx = torch.cat((idx[0::2], idx[1::2]))
            new_params = torch.zeros_like(shuffled_population)
            new_params[idx] = shuffled_population
            values = torch.zeros_like(policy_values)
            values[stacked_idx] = policy_values
            policies = torch.zeros_like(P_1)
            policies[stacked_idx] = P_1
            return new_params, values, policies
        
        self.params[idx] = shuffled_population



    def update_param_hybrid(
        self,
        lola: bool,
        lola_alpha: float,
        alpha: float,
        beta: float,
    ) -> None:
        """
        make an hybrid update with RL and imitation.
        Args:
            alpha (float): The weightage for RL updates.
            beta (float): The weightage for imitation updates.
        """
        rl_update, policy_values, policies = self.update_param_pg(alpha=alpha, lola_alpha=lola_alpha, lola=lola, return_update=True)

        imitating_partners = torch.multinomial(self.imitation_matrix, 1).squeeze(1)

        partner_values = policy_values[imitating_partners]
        
        # payoffs are used as imitation probabilities, so rescale them from [min, max] of the game to [0, 1]
        A_min, A_max = self.payoff_matrix_th.min(), self.payoff_matrix_th.max()
        partner_values = (partner_values - A_min) / (A_max - A_min)

        imitating_probabilities = torch.rand(self.size)

        imitation_change = torch.zeros_like(self.params)
        imitation_mask = imitating_probabilities < partner_values

        # imitation gradient is the difference between the partner's policy and the individual's policy if the imitation is successful.
        imitation_change[imitation_mask] = (
            policies[imitating_partners[imitation_mask]]
            - policies[imitation_mask]
        )

        imitation_update = torch.log1p(beta * imitation_change / policies)


        self.params = rl_update + imitation_update

        # change in the param space calculated as 
        # imitation_param_change = torch.log(imitation_change) 