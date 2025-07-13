"""Population simulation for imitation of success, weighted voter model."""

import numpy as np
import torch
from bandit import (
    BanditLinear,
    BanditSigmoid,
)  # Assuming Bandit class is defined in bandit.py


def get_pop_vector(
    pop_types: np.ndarray | torch.Tensor, n_types: int
) -> np.ndarray | torch.Tensor:
    """
    Computes the population vector from the population types.
    Args:
      pop_types (np.ndarray): Array of population types
    Returns:
      np.ndarray: Population vector indicating the proportion of each type in the population
    """
    if isinstance(pop_types, np.ndarray):
        unique_types, counts = np.unique(pop_types, return_counts=True)
        pop_vector = np.zeros(n_types)
        pop_vector[unique_types] = counts / len(pop_types)

    elif isinstance(pop_types, torch.Tensor):
        unique_types, counts = torch.unique(pop_types, return_counts=True)
        pop_vector = torch.zeros(n_types, device=pop_types.device)
        pop_vector[unique_types] = counts / len(pop_types)

    return pop_vector


def imitaton_of_success(
    steps: int, population_size: int, seeds: int, bandit: BanditLinear | BanditSigmoid
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the imitation of success.
    Args:
      steps (int): Number of steps to run the algorithm
      population_size (int): Size of the population of individuals
      seeds (int): Number of seeds for random initialization
      bandit (Bandit): The bandit environment to interact with
    Returns:
      mean_payoff (np.ndarray): Array indicating evolution of average payoffs over time
      optimal_type_ratio (np.ndarray): Array indicating evolution optimal type proportion over time
    """

    mean_payoff = np.zeros((seeds, steps))
    optimal_type_ratio = np.zeros((seeds, steps))

    for j, _ in enumerate(range(seeds)):
        pop_types = np.repeat(
            np.arange(bandit.n_action), population_size // bandit.n_action
        )
        # print(pop_types)
        individuals = np.arange(stop=population_size)  # individuals by ids
        for i, _ in enumerate(range(steps)):
            # get population vector
            pop_vector = get_pop_vector(pop_types=pop_types, n_types=bandit.n_action)
            # pop optimal actions
            optimal_type_ratio[j, i] = pop_vector[bandit.optimal_action()]
            # get payoffs for each individual
            payoffs = bandit.pull(pop_types)
            shuffled_individuals = individuals.copy()
            # shuffle individuals to imitate partners
            np.random.shuffle(shuffled_individuals)
            # partners to imitate
            imitating_partner = pop_types[shuffled_individuals]
            # rewards of partners
            imitating_reward = payoffs[shuffled_individuals]
            # random probabilities for imitation
            probabilities = np.random.rand(population_size)
            # update types based on imitation
            pop_types = np.where(
                probabilities < imitating_reward, imitating_partner, pop_types
            )
            mean_payoff[j, i] = np.mean(payoffs)  # average payoff

        # check if there is 1 in the policy vector
        # if not np.any(pop_vector == 1):
        #     print(
        #         f"Seed {j} did not converge to a single type for population size {population_size}."
        #     )
        #     print(f"Policy vector: {pop_vector}")

    return mean_payoff, optimal_type_ratio


def weighted_voter_rule(
    steps: int,
    population_size: int,
    seeds: int,
    bandit: BanditLinear | BanditSigmoid,
    neighbourhood_size: int,
    disjoint_neighbourhood: bool,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the weighted voter model.
    Args:
      steps (int): Number of steps to run the algorithm
      population_size (int): Size of the population of individuals
      seeds (int): Number of seeds for random initialization
      bandit (Bandit): The bandit environment to interact with
      neighbourhood_size (int): The neighbourhood size where the bees can broadcast and listen to
      disjoint_neighbourhood (bool): Should I have a disjoint neighbourhood or not
      device (str): cpu or cuda
    Returns:
      mean_qualities (np.ndarray): Array indicating evolution of average payoffs over time
      optimal_option_ratio (np.ndarray): Array indicating evolution optimal option proportion over time
    """
    mean_qualities = torch.zeros((seeds, steps), device=device)
    optimal_option_ratio = torch.zeros((seeds, steps), device=device)
    types = torch.arange(bandit.n_action)
    bees = torch.arange(0, population_size)

    # tiling bees
    bees_copy_tiled = bees.tile((population_size, 1))
    # These are possible neighbours, remove yourself from these neighbours
    bees_copy_tiled[bees, bees] = -1
    bees_copy_tiled = bees_copy_tiled[bees_copy_tiled != -1]
    bees_copy_tiled = bees_copy_tiled.reshape(population_size, -1)
    weights = torch.ones_like(bees_copy_tiled, dtype=torch.float)

    optimal_action_index_th = torch.tensor(bandit.optimal_action(), device=device)

    # create a one hot quality matrix for the bees

    pop_opinions_init = types.repeat(population_size // bandit.n_action)

    for j in range(seeds):

        # initialize population with equal distribution of types
        pop_opinions = pop_opinions_init

        for i in range(steps):
            # get population vector
            pop_vector = get_pop_vector(pop_types=pop_opinions, n_types=bandit.n_action)

            # pop optimal actions
            optimal_option_ratio[j, i] = pop_vector[optimal_action_index_th]
            # get qualities for each bees
            qualities = bandit.pull(pop_opinions)
            quality_matrix = torch.zeros(
                (population_size, bandit.n_action), dtype=torch.float32, device=device
            )
            # fill them with the qualities
            quality_matrix[bees, pop_opinions] = qualities
            # compute the qualities sum per option

            if disjoint_neighbourhood:
                # the idea is to create disjoint neighbourhoods so that you can have influence on exactly M neighbours.
                shuffled_bees = bees.copy()
                np.random.shuffle(shuffled_bees)
                shuffled_bees_in_nei = shuffled_bees.reshape(
                    int(population_size // neighbourhood_size), neighbourhood_size
                )

                # one hotted quality matrix
                quality_matrix_nei = quality_matrix[shuffled_bees_in_nei]
                quality_sum_per_option_nei = np.sum(quality_matrix_nei, axis=1)
                weighted_proportions_nei = np.einsum(
                    "ij,i->ij",
                    quality_sum_per_option_nei,
                    1 / np.sum(quality_sum_per_option_nei, axis=1),
                )

                weighted_proportions_nei_tensor = torch.tensor(weighted_proportions_nei)
                new_pop_opinions = torch.distributions.Categorical(
                    weighted_proportions_nei_tensor
                ).sample(sample_shape=(neighbourhood_size,))
                pop_opinions = new_pop_opinions.reshape(population_size).numpy()

            else:
                # sampling (neighbourhood_size - 1) per bee, so you can have influence on multiple bees at the same time.
                idx = torch.multinomial(
                    weights, num_samples=neighbourhood_size, replacement=False
                )
                neighbours = bees_copy_tiled.gather(1, idx)
                quality_matrix_nei = quality_matrix[
                    neighbours
                ]  # shape: (population_size , neighborhood_size, n_arms)
                quality_sum_per_option_nei = quality_matrix_nei.sum(dim=1)
                weighted_proportions_nei = (
                    quality_sum_per_option_nei
                    / quality_sum_per_option_nei.sum(dim=1, keepdim=True)
                )
                # weighted_proportions_nei = np.einsum(
                #     "ij,i->ij",
                #     quality_sum_per_option_nei,
                #     1 / np.sum(quality_sum_per_option_nei, axis=1),
                # )
                pop_opinions = torch.distributions.Categorical(
                    weighted_proportions_nei
                ).sample()

                # import pdb

                # pdb.set_trace()

            # naive implementation of the whole population as my neighbourhood
            # print(pop_opinions)
            # print(shuffled_bees_in_neighbourhood)
            # print(quality_sum_per_option_neighbourhood)
            # print(np.sum(quality_sum_per_option_neighbourhood, axis=1))
            # print(weighted_proportions_neighbourhood)

            # quality_sum_per_option = np.sum(quality_matrix, axis=0)
            # # weighted proportion which defines the  the distribution of
            # # votes cast for each type
            # weighted_proportions = quality_sum_per_option / np.sum(
            #     quality_sum_per_option
            # )
            # # new opinions of the bees based on the weighted proportions
            # pop_opinions = np.random.choice(
            #     np.arange(bandit.n_action), population_size, p=weighted_proportions
            # )
            # compute the mean quality for this step
            # print(i)
            mean_qualities[j, i] = qualities.mean()

        # check if there is 1 in the policy vector
        # if not np.any(pop_vector == 1):
        #     print(
        #         f"Seed {j} did not converge to a single type for"
        #         + f"population size {population_size}: Policy vector: {pop_vector}"
        #     )

    return mean_qualities.cpu().numpy(), optimal_option_ratio.cpu().numpy()
