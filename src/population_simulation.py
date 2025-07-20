"""Population simulation for imitation of success, weighted voter model."""

import numpy as np
import torch
from bandit import (
    BanditLinear,
    BanditSigmoid,
)  # Assuming Bandit class is defined in bandit.py

import os
import torch
import multiprocessing as mp
from functools import partial
import time

torch.set_num_threads(1)

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS (used by NumPy)
os.environ["MKL_NUM_THREADS"] = "1"  # MKL (used by NumPy on Intel)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr if used


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
    steps: int,
    population_size: int,
    iterations: int,
    bandit: BanditLinear | BanditSigmoid,
    name: str,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the imitation of success.
    Args:
      steps (int): Number of steps to run the algorithm
      population_size (int): Size of the population of individuals
      iterations (int): Number of iterations for random initialization
      bandit (Bandit): The bandit environment to interact with
      name (str): name of the scenario for the bandit
      device (str): should be it on cpu or gpu
    Returns:
      mean_payoff (np.ndarray): Array indicating evolution of average payoffs over time
      optimal_type_ratio (np.ndarray): Array indicating evolution optimal type proportion over time
    """

    mean_payoff = torch.zeros((iterations, steps), device=device)
    optimal_type_ratio = torch.zeros((iterations, steps), device=device)

    seed = int(time.time() * 1e6) % (2**32 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    bandit = BanditLinear(name=name, device=device)

    for j, _ in enumerate(range(iterations)):

        types = torch.arange(bandit.n_action)
        individuals = torch.arange(0, population_size)

        # tiling bees
        individuals_copy_tiled = individuals.tile((population_size, 1))
        # These are possible neighbours, remove yourself from these neighbours
        individuals_copy_tiled[individuals, individuals] = -1
        individuals_copy_tiled = individuals_copy_tiled[individuals_copy_tiled != -1]
        individuals_copy_tiled = individuals_copy_tiled.reshape(population_size, -1)
        weights = torch.ones_like(individuals_copy_tiled, dtype=torch.float)

        optimal_action_index_th = torch.tensor(bandit.optimal_action(), device=device)

        # create a one hot quality matrix for the bees

        pop_types_init = types.repeat(population_size // bandit.n_action)
        # initialize population with equal distribution of types
        pop_types = pop_types_init

        for i, _ in enumerate(range(steps)):
            # get population vector
            pop_vector = get_pop_vector(pop_types=pop_types, n_types=bandit.n_action)
            # pop optimal actions
            optimal_type_ratio[j, i] = pop_vector[optimal_action_index_th]
            # get payoffs for each individual
            payoffs = bandit.pull(pop_types)
            # get imitating p
            idx = torch.distributions.Categorical(weights).sample()
            imitating_partner_idx = individuals_copy_tiled.gather(1, idx)
            # partners to imitate
            imitating_partner_type = pop_types[imitating_partner_idx]
            # rewards of partners
            imitating_reward = payoffs[imitating_partner_idx]
            # random probabilities for imitation
            probabilities = torch.rand(population_size)
            # update types based on imitation
            pop_types = torch.where(
                probabilities < imitating_reward, imitating_partner_type, pop_types
            )
            mean_payoff[j, i] = payoffs.mean()  # average payoff

        # check if there is 1 in the policy vector
        # if not np.any(pop_vector == 1):
        #     print(
        #         f"Seed {j} did not converge to a single type for population size {population_size}."
        #     )
        #     print(f"Policy vector: {pop_vector}")

    return mean_payoff.cpu().numpy(), optimal_type_ratio.cpu().numpy()


def majority_rule(
    steps: int,
    population_size: int,
    iterations: int,
    neighbourhood_size: int,
    disjoint_neighbourhood: bool,
    number_of_votes: int,
    use_neighbourhood: bool,
    name: str,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the majority rule model.
    Args:
      steps (int): Number of steps to run the algorithm
      population_size (int): Size of the population of individuals
      iterations (int): Number of iterations for random initialization
      neighbourhood_size (int): The neighbourhood size where the bees can broadcast and listen to.
      number_of_votes: Number of votes to look at before taking a decision (apply majority rule).
      disjoint_neighbourhood (bool): Should I have a disjoint neighbourhood or not
      use_neighbourhood (bool): use neighbourhood or use a simple
      name (str): scenario for bandit
      device (str): cpu or cuda
    Returns:
      mean_qualities (np.ndarray): Array indicating evolution of average payoffs over time
      optimal_option_ratio (np.ndarray): Array indicating evolution optimal option proportion over time
    """
    mean_qualities = torch.zeros((iterations, steps), device=device)
    optimal_option_ratio = torch.zeros((iterations, steps), device=device)
    seed = int(time.time() * 1e6) % (2**32 - 1)
    # print(f"seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    bandit = BanditLinear(name=name, device=device)
    for j in range(iterations):

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
        # initialize population with equal distribution of types
        pop_opinions = pop_opinions_init

        for i in range(steps):
            # print(pop_opinions)
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

            if use_neighbourhood:
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

                    weighted_proportions_nei_tensor = torch.tensor(
                        weighted_proportions_nei
                    )
                    new_pop_opinions = torch.distributions.Categorical(
                        weighted_proportions_nei_tensor
                    ).sample(sample_shape=(neighbourhood_size,))
                    pop_opinions = new_pop_opinions.reshape(population_size).numpy()

                else:
                    # sampling (neighbourhood_size - 1) per bee, so you can have influence on multiple bees at the same time.
                    if neighbourhood_size == population_size:
                        neighbours = bees_copy_tiled
                    else:
                        idx = torch.multinomial(
                            weights, num_samples=neighbourhood_size, replacement=False
                        )
                        neighbours = bees_copy_tiled.gather(1, idx)

                    # one hotted quality matrix
                    quality_matrix_nei = quality_matrix[neighbours]
                    quality_sum_per_option_nei = quality_matrix_nei.sum(dim=1)
                    weighted_proportions_nei = (
                        quality_sum_per_option_nei
                        / quality_sum_per_option_nei.sum(dim=1, keepdim=True)
                    )

                    # if there need to be consideration of too many votes, just take an argmax
                    if number_of_votes >= 100000:
                        max_types_mask = weighted_proportions_nei.max(
                            dim=1, keepdim=True
                        ).values
                        majority_proportion = (
                            weighted_proportions_nei == max_types_mask
                        ).int()
                        majority_proportion = (
                            majority_proportion
                            / majority_proportion.sum(dim=1, keepdim=True)
                        )

                    else:
                        votes = (
                            torch.distributions.Categorical(weighted_proportions_nei)
                            .sample(sample_shape=(number_of_votes, 1))
                            .reshape(population_size, number_of_votes)
                        )

                        counts = torch.nn.functional.one_hot(votes).sum(dim=1)

                        max_types_mask = counts.max(dim=1, keepdim=True).values

                        majority_proportion = (counts == max_types_mask).int()

                        majority_proportion = (
                            majority_proportion
                            / majority_proportion.sum(dim=1, keepdim=True)
                        )

                    pop_opinions = torch.distributions.Categorical(
                        majority_proportion
                    ).sample()

            else:
                quality_sum_per_option = quality_matrix.sum(dim=0)
                # weighted proportion which defines the  the distribution of
                # votes cast for each type
                weighted_proportions = (
                    quality_sum_per_option / quality_sum_per_option.sum(dim=0)
                )
                # new opinions of the bees based on the weighted proportions
                pop_opinions = torch.distributions.Categorical(
                    weighted_proportions
                ).sample(sample_shape=(population_size,))

            # compute the mean quality for this step
            # print(i)
            mean_qualities[j, i] = qualities.mean()

        # check if there is 1 in the policy vector
        # if not torch.any(pop_vector == 1):
        #     print(
        #         f"Seed {j} did not converge to a single type for"
        #         + f"population size {population_size}: Policy vector: {pop_vector}"
        #     )

    return mean_qualities.cpu().numpy(), optimal_option_ratio.cpu().numpy()


def weighted_voter_rule(
    steps: int,
    population_size: int,
    iterations: int,
    neighbourhood_size: int,
    disjoint_neighbourhood: bool,
    use_neighbourhood: bool,
    name: str,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the weighted voter model.
    Args:
      steps (int): Number of steps to run the algorithm
      population_size (int): Size of the population of individuals
      iterations (int): Number of iterations for random initialization
      neighbourhood_size (int): The neighbourhood size where the bees can broadcast and listen to
      disjoint_neighbourhood (bool): Should I have a disjoint neighbourhood or not
      use_neighbourhood (bool): use neighbourhood or use a simple
      name (str): scenario for bandit
      device (str): cpu or cuda
    Returns:
      mean_qualities (np.ndarray): Array indicating evolution of average payoffs over time
      optimal_option_ratio (np.ndarray): Array indicating evolution optimal option proportion over time
    """
    mean_qualities = torch.zeros((iterations, steps), device=device)
    optimal_option_ratio = torch.zeros((iterations, steps), device=device)

    for j in range(iterations):
        seed = int(time.time() * 1e6) % (2**32 - 1)
        # print(f"seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        bandit = BanditLinear(name=name, device=device)

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

            if use_neighbourhood:
                if disjoint_neighbourhood:
                    # the idea is to create disjoint neighbourhoods so that you can have influence on exactly
                    # neighbourhood_size neighbours.
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

                    weighted_proportions_nei_tensor = torch.tensor(
                        weighted_proportions_nei
                    )
                    new_pop_opinions = torch.distributions.Categorical(
                        weighted_proportions_nei_tensor
                    ).sample(sample_shape=(neighbourhood_size,))
                    pop_opinions = new_pop_opinions.reshape(population_size).numpy()

                else:
                    # sampling (neighbourhood_size - 1) per bee, so you can have influence on multiple bees at the same time.
                    if neighbourhood_size == population_size:
                        neighbours = bees_copy_tiled
                    else:
                        idx = torch.multinomial(
                            weights, num_samples=neighbourhood_size, replacement=False
                        )
                        neighbours = bees_copy_tiled.gather(1, idx)
                    quality_matrix_nei = quality_matrix[neighbours]
                    quality_sum_per_option_nei = quality_matrix_nei.sum(dim=1)
                    weighted_proportions_nei = (
                        quality_sum_per_option_nei
                        / quality_sum_per_option_nei.sum(dim=1, keepdim=True)
                    )
                    pop_opinions = torch.distributions.Categorical(
                        weighted_proportions_nei
                    ).sample()

            else:
                quality_sum_per_option = quality_matrix.sum(dim=0)
                # weighted proportion which defines the the distribution of
                # votes cast for each type
                weighted_proportions = (
                    quality_sum_per_option / quality_sum_per_option.sum(dim=0)
                )
                # new opinions of the bees based on the weighted proportions
                pop_opinions = torch.distributions.Categorical(
                    weighted_proportions
                ).sample(sample_shape=(population_size,))

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


def run_parallel_simulation_wvr(
    steps: int,
    population_size: int,
    seeds: int,
    neighbourhood_size: int,
    disjoint_neighbourhood: bool,
    use_neighbourhood: bool,
    name: str,
    n_proc: int,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:

    iterations = int(seeds // n_proc)
    args_list = [
        (
            steps,
            population_size,
            iterations,
            neighbourhood_size,
            disjoint_neighbourhood,
            use_neighbourhood,
            name,
            device,
        )
        for _ in range(n_proc)
    ]

    with mp.Pool(processes=n_proc) as pool:
        results = pool.starmap(weighted_voter_rule, args_list)

    all_mean_qualities, all_opt_ratios = zip(*results)
    all_mean_qualities = np.stack(all_mean_qualities)
    all_opt_ratios = np.stack(all_opt_ratios)

    return all_mean_qualities.reshape(seeds, steps), all_opt_ratios.reshape(
        seeds, steps
    )


def run_parallel_simulation_majority(
    steps: int,
    population_size: int,
    seeds: int,
    neighbourhood_size: int,
    disjoint_neighbourhood: bool,
    number_of_votes: int,
    use_neighbourhood: bool,
    name: str,
    n_proc: int,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:

    iterations = int(seeds // n_proc)
    args_list = [
        (
            steps,
            population_size,
            iterations,
            neighbourhood_size,
            disjoint_neighbourhood,
            number_of_votes,
            use_neighbourhood,
            name,
            device,
        )
        for _ in range(n_proc)
    ]

    with mp.Pool(processes=n_proc) as pool:
        results = pool.starmap(majority_rule, args_list)

    all_mean_qualities, all_opt_ratios = zip(*results)
    all_mean_qualities = np.stack(all_mean_qualities)
    all_opt_ratios = np.stack(all_opt_ratios)

    return all_mean_qualities.reshape(seeds, steps), all_opt_ratios.reshape(
        seeds, steps
    )


def run_parallel_simulation_is(
    steps: int,
    population_size: int,
    iterations: int,
    name: str,
    n_proc: int,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:

    iterations_per_process = int(iterations // n_proc)
    args_list = [
        (
            steps,
            population_size,
            iterations_per_process,
            name,
            device,
        )
        for _ in range(n_proc)
    ]

    with mp.Pool(processes=n_proc) as pool:
        results = pool.starmap(imitaton_of_success, args_list)

    all_mean_qualities, all_opt_ratios = zip(*results)
    all_mean_qualities = np.stack(all_mean_qualities)
    all_opt_ratios = np.stack(all_opt_ratios)

    return all_mean_qualities.reshape(iterations, steps), all_opt_ratios.reshape(
        iterations, steps
    )
