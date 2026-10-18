
import numpy as np
import torch
import matplotlib.pyplot as plt


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

# Copied from Fast_LOLA_TMLR/RL-societies/rl_societies.ipynb (plot_3d_experiment_3d_lines),
# with the changes marked below
def plot_3d_experiment_3d_lines(
    action_1, action_2, action_3, hist, title=None, azim=None, dpi=200
):
    with torch.no_grad():
        mat_3d_to_2d = torch.Tensor([[0, 1, 1 / 2], [0, 0, np.sqrt(3) / 2]])

        # Create canvas
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(
            projection="3d"
        )  # changed: Axes3D(fig, auto_add_to_figure=False) fails on matplotlib >= 3.6
        ax.view_init(azim=azim)

        # Constants
        # v11 = torch.tensor([1, 0, 0], dtype=torch.float)
        # v12 = torch.tensor([0, 1, 0], dtype=torch.float)
        # v13 = torch.tensor([0, 0, 1], dtype=torch.float)

        # v21 = torch.tensor([1, 1, 0], dtype=torch.float)
        # v22 = torch.tensor([1, 0, 1], dtype=torch.float)
        # v23 = torch.tensor([0, 1, 1], dtype=torch.float)

        c = "lightgrey"
        linewidth = 1
        plt.plot([0, 1], [0, 0], c=c, linewidth=linewidth)
        plt.plot([0, 1 / 2], [0, np.sqrt(3) / 2], c=c, linewidth=linewidth)
        plt.plot([1 / 2, 1], [np.sqrt(3) / 2, 0], c=c, linewidth=linewidth)
        # plt.axis('off')

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_zlabel("\n\npopulation step\n")

        up_offset = 0
        fontsize = 10
        ax.text(
            0,
            0,
            0 + up_offset,
            action_1,
            horizontalalignment="center",
            fontsize=fontsize,
        )
        ax.text(
            1,
            0,
            0 + up_offset,
            action_2,
            horizontalalignment="center",
            fontsize=fontsize,
        )
        ax.text(
            1 / 2,
            np.sqrt(3) / 2,
            0 + up_offset,
            action_3,
            horizontalalignment="center",
            fontsize=fontsize,
        )

        hist_t = torch.tensor(np.array(hist))
        print(hist_t.shape)

        policies_hist = torch.nn.functional.softmax(hist_t, dim=2)
        policies_hist_2d = torch.einsum("ij,hbj->hbi", mat_3d_to_2d, policies_hist)
        z = np.array(range(len(hist)))
        nb_agents = policies_hist_2d.shape[1]
        for agent in range(nb_agents):
            x = policies_hist_2d[:, agent, 0]
            y = policies_hist_2d[:, agent, 1]
            ax.plot3D(x, y, z)

        # added: population-mean policy
        mean_2d = policies_hist_2d.mean(dim=1)
        ax.plot3D(mean_2d[:, 0], mean_2d[:, 1], z, c="k", linewidth=2.5, label="mean")
        ax.legend()

        if title:
            plt.title(title)
        ax.set_box_aspect(aspect=None, zoom=0.8)
        # removed: plt.savefig(SAV_DIR / f"{title}.png", bbox_inches='tight')
        plt.show()