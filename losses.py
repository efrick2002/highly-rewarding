from utils import get_registry_decorator
from typing import Dict, Callable
import torch
from torch.nn import functional as F
import os

RANK = int(os.environ.get("RANK", -1))

REGISTERED_LOSSES: Dict[str, Callable] = {}

register = get_registry_decorator(REGISTERED_LOSSES)

@register("bt-pairwise-reward")
def pairwise_reward_loss(output: Dict, labels: torch.Tensor, num_items_in_batch=None) -> torch.Tensor:
    """
    Compute the pairwise reward loss.
    """
    # print(labels)
    # print(f"Rank {RANK}: num_items_in_batch: {num_items_in_batch}")

    num_items_in_batch = num_items_in_batch // 2

    # print(f"Rank {RANK}: Actual num_items_in_batch: {num_items_in_batch}")

    rewards: torch.Tensor = output["rewards"].float()

    # print(f"Rank {RANK}: Rewards: {rewards}")

    # The labels are of shape (bs, 2). The second dimension indicates the order of the two messages. We now want to index into the rewards
    # tensor with the labels so we can get the first column as the winner reward, and the second column as the loser reward.
    winner_rewards = rewards[torch.arange(rewards.shape[0]), labels[:, 0]]
    loser_rewards = rewards[torch.arange(rewards.shape[0]), labels[:, 1]]

    # Now we compute the difference between the winner and loser rewards.
    diff = winner_rewards - loser_rewards

    # Now we compute the losses
    losses = -F.logsigmoid(diff)

    loss = losses.sum()

    loss = loss / num_items_in_batch

    return loss


# @register("grk-pairwise-reward")
# def grk_pairwise_reward_loss(output: Dict, labels: torch.Tensor, num_items_in_batch=None) -> torch.Tensor:
#     """
#     Compute the grk pairwise reward loss.
#     """

#     num_items_in_batch = num_items_in_batch // 2

#     rewards: torch.Tensor = output["rewards"].float()
#     thetas: torch.Tensor = output["thetas"].float()

#     # The labels are of shape (bs, 2). The second dimension indicates the order of the two messages. We now want to index into the rewards
#     # tensor with the labels so we can get the first column as the winner reward, and the second column as the loser reward.
#     winner_rewards = rewards[torch.arange(rewards.shape[0]), labels[:, 0]]
#     loser_rewards = rewards[torch.arange(rewards.shape[0]), labels[:, 1]]

    




def thurstonian_loss(mu1, logvar1, mu2, logvar2):
    """
    mu1, logvar1: [batch_size] each, for the 'preferred' response
    mu2, logvar2: [batch_size] each, for the 'non-preferred' response
    
    We want to maximize p(r1 > r2) = Φ((mu1 - mu2) / sqrt(sigma1^2 + sigma2^2)).
    Minimizing the negative log-likelihood: -log p(r1 > r2).
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    
    # difference standard deviation: sqrt(var1 + var2)
    denom = torch.sqrt(var1 + var2 + 1e-8)
    z = (mu1 - mu2) / denom
    
    # Use torch.special.log_ndtr for a numerically stable computation of log(Φ(z))
    log_cdf = torch.special.log_ndtr(z)
    
    # negative log-likelihood
    nll = -log_cdf
    return nll.mean()


@register("thurstone-pairwise-reward")
def thurstone_pairwise_reward_loss(output: Dict, labels: torch.Tensor, num_items_in_batch=None) -> torch.Tensor:
    """
    Compute the thurstone pairwise reward loss.
    """

    num_items_in_batch = num_items_in_batch // 2

    means: torch.Tensor = output["means"].float()
    logvars: torch.Tensor = output["logvars"].float()
    
    # The labels are of shape (bs, 2). The second dimension indicates the order of the two messages. We now want to index into the rewards
    # tensor with the labels so we can get the first column as the winner reward, and the second column as the loser reward.
    winner_means = means[torch.arange(means.shape[0]), labels[:, 0]]
    loser_means = means[torch.arange(means.shape[0]), labels[:, 1]]

    winner_logvars = logvars[torch.arange(logvars.shape[0]), labels[:, 0]]
    loser_logvars = logvars[torch.arange(logvars.shape[0]), labels[:, 1]]

    loss = thurstonian_loss(winner_means, winner_logvars, loser_means, loser_logvars)

    loss = loss / num_items_in_batch

    return loss


