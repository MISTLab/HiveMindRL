"""Population simulation for imitation of success, weighted voter model."""

import numpy as np
from bandit import Bandit  # Assuming Bandit class is defined in bandit.py


def get_pop_vector(pop_types: np.ndarray, bandit: Bandit) -> np.ndarray:
    """
    Computes the population vector from the population types.
    Args:
      pop_types (np.ndarray): Array of population types
    Returns:
      np.ndarray: Population vector indicating the proportion of each type in the population
    """
    unique_types, counts = np.unique(pop_types, return_counts=True)
    pop_vector = np.zeros(bandit.n_action)
    pop_vector[unique_types] = counts / len(pop_types)
    return pop_vector


def imitaton_of_success(
    steps: int, population_size: int, seeds: int, bandit: Bandit
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
        pop_vector = np.ones(bandit.n_action) / bandit.n_action
        pop_types = np.random.choice(bandit.n_action, population_size, p=pop_vector)
        # print(pop_types)
        individuals = np.arange(stop=population_size)  # individuals by ids
        for i, _ in enumerate(range(steps)):
            # get population vector
            pop_vector = get_pop_vector(pop_types=pop_types, bandit=bandit)
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
            imitating_reward = payoffs.numpy()[shuffled_individuals]
            # random probabilities for imitation
            probabilities = np.random.rand(population_size)
            # update types based on imitation
            pop_types = np.where(
                probabilities < imitating_reward, imitating_partner, pop_types
            )
            mean_payoff[j, i] = np.mean(payoffs.numpy())  # average payoff

        # check if there is 1 in the policy vector
        # if not np.any(pop_vector == 1):
        #     print(
        #         f"Seed {j} did not converge to a single type for population size {population_size}."
        #     )
        #     print(f"Policy vector: {pop_vector}")

    return mean_payoff, optimal_type_ratio


def weighted_voter_rule(
    steps: int, population_size: int, seeds: int, bandit: Bandit
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the weighted voter model.
    Args:
      steps (int): Number of steps to run the algorithm
      population_size (int): Size of the population of individuals
      seeds (int): Number of seeds for random initialization
      bandit (Bandit): The bandit environment to interact with
    Returns:
      mean_qualities (np.ndarray): Array indicating evolution of average payoffs over time
      optimal_option_ratio (np.ndarray): Array indicating evolution optimal option proportion over time
    """
    mean_qualities = np.zeros((seeds, steps))
    optimal_option_ratio = np.zeros((seeds, steps))

    for j in range(seeds):
        # initialize population with equal distribution of types
        pop_opinions = np.repeat(
            np.arange(bandit.n_action), population_size // bandit.n_action
        )
        bees = np.arange(0, population_size)
        for i in range(steps):
            # get population vector
            pop_vector = get_pop_vector(pop_types=pop_opinions, bandit=bandit)
            # pop optimal actions
            optimal_option_ratio[j, i] = pop_vector[bandit.optimal_action()]
            # get qualities for each bees
            qualities = bandit.pull(pop_opinions)
            # create a quality matrix for the bees
            quality_matrix = np.zeros((population_size, bandit.n_action))
            # fill them with the qualities
            quality_matrix[bees, pop_opinions] = qualities.numpy()
            # compute the qualities sum per option
            quality_sum_per_option = np.sum(quality_matrix, axis=0)
            # weighted proportion which defines the  the distribution of
            # votes cast for each type
            weighted_proportions = quality_sum_per_option / np.sum(
                quality_sum_per_option
            )
            # new opinions of the bees based on the weighted proportions
            pop_opinions = np.random.choice(
                np.arange(bandit.n_action), population_size, p=weighted_proportions
            )
            # compute the mean quality for this step
            mean_qualities[j, i] = np.mean(qualities.numpy())

        # check if there is 1 in the policy vector
        # if not np.any(pop_vector == 1):
        #     print(
        #         f"Seed {j} did not converge to a single type for"
        #         + f"population size {population_size}: Policy vector: {pop_vector}"
        #     )

    return mean_qualities, optimal_option_ratio
