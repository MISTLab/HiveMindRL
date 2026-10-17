
import numpy as np
import torch


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

def init_logits(N, A, r, seed=0):                                                                                                                      
        """Logits for N agents, uniform over a disk of radius r around zero (the uniform policy)."""                                                       
        gen = torch.Generator().manual_seed(seed)                                                                                                          
        v = torch.randn(N, A, generator=gen)                                                                                                               
        v -= v.mean(dim=-1, keepdim=True)          # stay in the plane softmax responds to                                                                 
        v /= v.norm(dim=-1, keepdim=True)          # random unit direction                                                                                 
        radius = r * torch.rand(N, 1, generator=gen) ** (1 / (A - 1))   # uniform over the disk                                                            
        return radius * v

def shuffle_agents(population, nb_agent_pairs, device):
    idx = torch.randperm(nb_agent_pairs * 2).to(device)
    shuffled_population = population[idx]
    return shuffled_population, idx