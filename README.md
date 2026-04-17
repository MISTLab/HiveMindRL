#  HiveMindRL 🐝🤖


![](assets/intro.png)

## Abstract ✨

Decision-making is an essential attribute of any intelligent agent or group.
Natural systems are known to converge to optimal strategies through at least two distinct mechanisms: collective decision-making via imitation of others, and individual trial-and-error.
This paper establishes an equivalence between these two paradigms by drawing from the well-established collective decision-making model of nest-hunting in swarms of honey bees.
We show that the emergent distributed cognition (sometimes referred to as the $\textit{hive mind}$) arising from individual bees following simple, local imitation-based rules is that of a single online reinforcement learning (RL) agent interacting with many parallel environments.
The update rule through which this macro-agent learns is a bandit algorithm that we coin $\textit{Maynard-Cross Learning}$.
Our analysis implies that a group of cognition-limited organisms can be equivalent to a more complex, reinforcement-enabled entity, substantiating the idea that group-level intelligence may explain how seemingly simple and blind individual behaviors are selected in nature.

From a biological perspective, this analysis suggests how such imitation strategies
evolved: they constitute a scalable form of reinforcement learning at the group
level, aligning with theories of kin and group selection. Beyond biology, the
framework offers new tools for analyzing economic and social systems where
individuals imitate successful strategies, effectively participating in a
collective learning process. In swarm intelligence, our findings will inform the
design of scalable collective systems in artificial domains, enabling
RL-inspired mechanisms for coordination and adaptability at scale.



banditcongestion = BanditCongestion(congestion_factors=[0.1, 0.9], device="cpu")

policy_vector = np.array([0.5, 0.5])
actions = np.random.choice(
    banditcongestion.return_no_actions(), size=1000, p=policy_vector
)

print(policy_vector[actions])
print(banditcongestion.congestion_factors[actions])

rewards = banditcongestion.pull(actions, policy_vector)

# plot rewards distribution per action type
rewards_action_0 = rewards[actions == 0]
rewards_action_1 = rewards[actions == 1]

# plot them side by side
fig, ax = plt.subplots(1, 2)
ax[0].hist(rewards_action_0, bins=20)
ax[1].hist(rewards_action_1, bins=20)
# plt.hist(rewards_action_0, bins=20)
ax[0].set_xlim(0, 1)
ax[1].set_xlim(0, 1)

mean_fitness, policy = replicator_dynamic(
    delta=1, bandit=banditcongestion, steps=30, trd=False
)

# plot mean fitness
fig, ax = plt.subplots(2, 1)
ax[0].plot(mean_fitness)

ax[1].plot(policy)
ax[1].plot(1 - policy)
ax[1].set_ylim(0, 1)

steps = 100
rewards, policy_cl = parallel_cross_learning(
    steps=steps, seeds=10, bandit=banditcongestion, parallel_envs=1000
)

mean_fitness_rd, policy_rd = replicator_dynamic(
    delta=1, bandit=banditcongestion, steps=steps, trd=False
)
