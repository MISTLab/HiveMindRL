from population_simulation import (
    weighted_voter_rule,
    majority_rule,
    imitaton_of_success,
)
from bandit import BanditLinear


if __name__ == "__main__":

    steps = 100
    population_size = 10
    iterations = 1
    neighbourhood_size = 2
    disjoint_neighbourhood = False
    number_of_votes = 10
    use_neighbourhood = True
    name = "evenly spaced"
    device = "cpu"
    bandit = BanditLinear(name="evenly spaced")
    deterministic = True
    switch = "is_stoc"
    stop_if_end = True
    print(bandit.q_star)

    _, _ = majority_rule(
        steps=steps,
        population_size=population_size,
        iterations=iterations,
        neighbourhood_size=neighbourhood_size,
        disjoint_neighbourhood=disjoint_neighbourhood,
        number_of_votes=number_of_votes,
        use_neighbourhood=use_neighbourhood,
        name=name,
        device=device,
        stop_if_end=stop_if_end,
    )

    # _, _ = weighted_voter_rule(
    #     steps=steps,
    #     population_size=population_size,
    #     iterations=iterations,
    #     disjoint_neighbourhood=disjoint_neighbourhood,
    #     use_neighbourhood=use_neighbourhood,
    #     neighbourhood_size=neighbourhood_size,
    #     device=device,
    #     name=name,
    #     switch=switch,
    # )

    # _, _ = imitaton_of_success(
    #     steps=steps,
    #     population_size=population_size,
    #     iterations=seeds,
    #     name=name,
    #     device=device,
    #     deterministic=True,
    # )
