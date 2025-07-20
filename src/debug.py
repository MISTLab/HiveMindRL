from population_simulation import weighted_voter_rule, majority_rule
from bandit import BanditLinear


if __name__ == "__main__":

    steps = 100
    population_size = 10
    seeds = 1
    neighbourhood_size = 2
    disjoint_neighbourhood = False
    number_of_votes = 10
    use_neighbourhood = True
    name = "evenly spaced"
    device = "cpu"
    bandit = BanditLinear(name="evenly spaced")
    print(bandit.q_star)

    _, _ = majority_rule(
        steps=steps,
        population_size=population_size,
        iterations=seeds,
        neighbourhood_size=neighbourhood_size,
        disjoint_neighbourhood=disjoint_neighbourhood,
        number_of_votes=number_of_votes,
        use_neighbourhood=use_neighbourhood,
        name=name,
        device=device,
    )
