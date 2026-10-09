import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFeedForward(nn.Module):
    """
    Sparse top-1 Mixture-of-Experts feed-forward block (Switch Transformer
    style), as a drop-in replacement for the dense linear1 -> activation ->
    linear2 block in EBTransformer.

    Each token is routed to exactly one of `num_experts` small FFNs by a
    learned router; the chosen expert's output is scaled by the router's
    softmax probability for that expert (the Switch Transformer gate).
    `last_aux_loss` holds the load-balancing auxiliary loss from the most
    recent forward call, which the caller is responsible for adding to the
    training loss (it does nothing on its own).
    """

    def __init__(self, dmodel, d_ff, num_experts, activation="gelu", device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(dmodel, num_experts, bias=False, **factory_kwargs)
        self.experts_fc1 = nn.ModuleList(
            [nn.Linear(dmodel, d_ff, **factory_kwargs) for _ in range(num_experts)]
        )
        self.experts_fc2 = nn.ModuleList(
            [nn.Linear(d_ff, dmodel, **factory_kwargs) for _ in range(num_experts)]
        )
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.last_aux_loss = None

    def forward(self, x):
        """
        Args:
            x: B x N x dmodel
        Returns:
            out: B x N x dmodel
        """
        B, N, D = x.shape
        logits = self.router(x)                       # B x N x E
        probs = torch.softmax(logits, dim=-1)          # B x N x E
        gate, expert_idx = probs.max(dim=-1)            # B x N (top-1 routing)

        flat_x = x.reshape(-1, D)                        # (B*N) x D
        flat_idx = expert_idx.reshape(-1)                 # (B*N,)
        flat_gate = gate.reshape(-1)                      # (B*N,)
        out = torch.zeros_like(flat_x)

        for e in range(self.num_experts):
            mask = flat_idx == e
            if mask.any():
                h = self.activation(self.experts_fc1[e](flat_x[mask]))
                out[mask] = self.experts_fc2[e](h)

        out = out * flat_gate.unsqueeze(-1)
        out = out.reshape(B, N, D)

        # Switch Transformer load-balancing loss: num_experts * sum_e f_e * P_e,
        # where f_e = fraction of tokens routed to expert e and P_e = mean
        # router probability mass on e. Minimized (at 1.0) when routing is
        # uniform across experts; penalizes collapse onto a few experts.
        one_hot = F.one_hot(flat_idx, num_classes=self.num_experts).to(probs.dtype)
        f = one_hot.mean(dim=0)
        P = probs.reshape(-1, self.num_experts).mean(dim=0)
        self.last_aux_loss = self.num_experts * torch.sum(f * P)

        return out
