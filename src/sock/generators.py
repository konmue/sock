import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Generator(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_cond_steps: int = 1,
        noise_dim: int = None,  # defaults to noise_dim = dim
        initial_noise_dim: int = None,  # defaults to noise_dim
    ):
        super().__init__()
        self.dim = dim
        self.noise_dim = noise_dim or dim
        self.initial_noise_dim = initial_noise_dim or self.noise_dim

        # NOTE: when n_lags is large and the dataset is small, this likely overfits
        self.initial_state_generator = nn.Sequential(
            nn.Linear(dim * n_cond_steps + self.initial_noise_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        with torch.no_grad():
            self.initial_state_generator[-1].bias.zero_()
            self.initial_state_generator[-1].weight.mul_(1e-2)

        self.proj_in = nn.Linear(self.noise_dim, 2 * hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )
        nn.init.constant_(self.gate[1].bias, -2.0)

        self.proj_out = nn.Linear(hidden_dim, dim)
        with torch.no_grad():
            self.proj_out.bias.zero_()
            self.proj_out.weight.mul_(0.5)

    def forward(self, x0: Tensor, n_steps: int):
        bs = x0.size(0)
        device = x0.device
        initial_noise = torch.randn((bs, self.initial_noise_dim), device=device)

        h0_in = torch.cat((x0.flatten(start_dim=1), initial_noise), dim=-1)
        h0 = self.initial_state_generator(h0_in)

        z = torch.randn((bs, n_steps, self.noise_dim), device=device)
        z, zkip = self.proj_in(z).chunk(2, dim=-1)
        z = F.silu(z)
        h, _ = self.rnn(z, h0.unsqueeze(0))
        h = h + self.alpha * self.gate(h) * zkip
        x = self.proj_out(h)
        return torch.cat((x0, x), dim=1)
