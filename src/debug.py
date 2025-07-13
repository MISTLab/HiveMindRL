from population_simulation import weighted_voter_rule
from bandit import BanditLinear


if __name__ == "__main__":

    steps = 100
    population_size = 100
    seeds = 1
    neighbourhood_size = 10
    bandit = BanditLinear(name="evenly spaced")

    _, _ = weighted_voter_rule(
        steps=steps,
        population_size=population_size,
        seeds=seeds,
        bandit=bandit,
        neighbourhood_size=neighbourhood_size,
        disjoint_neighbourhood=False,
    )
