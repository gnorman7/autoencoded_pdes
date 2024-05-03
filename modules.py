import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional
import utils

#%% Standard / Basics
class SimpleFNN(nn.Module):
    def __init__(self, layers, activation):
        super(SimpleFNN, self).__init__()
        self.layers = layers
        self.activation = activation
        self.fc_layers = nn.ModuleList()

        # Create fully connected layers
        for i in range(len(layers) - 1):
            self.fc_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.fc_layers.append(activation)

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x

# 1D dim convolution network (multiple layers)
# UNTESTED
class SimpleCNN(nn.Module):
    def __init__(self, layers, activation, kernel_size=5, padding=1):
        super(SimpleCNN, self).__init__()
        self.layers = layers
        self.activation = activation
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv_layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.conv_layers.append(nn.Conv1d(layers[i], layers[i+1],
                                              kernel_size=self.kernel_size, padding=self.padding))
            if i < len(layers) - 2:
                self.conv_layers.append(activation)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

        self.mean_layer = nn.Linear(self.latent_dim, self.latent_dim)
        self.log_var_layer = nn.Linear(self.latent_dim, self.latent_dim)

    def encode(self, x: torch.Tensor):
        # Encode the input as N(mu, sigma^2 * I) = q(z|x). This is a distribution over z, so mu and sigma are same dim.
        # The so called "encoder" doesn't quite output these parameters, so we add mean_layer and log_var_layer
        x = self.encoder(x)
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        return mean, log_var

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor):
        # Here, rather than sampling from N(mu, sigma^2), we sample from N(0, 1), then scale and shift by mu and sigma
        # Sampling operation doesn't have AD, so we do this manually with operations that do
        eps = torch.randn_like(log_var)
        z = mean + torch.exp(0.5 * log_var) * eps
        return z

    def loss_fn(self, x: torch.Tensor, beta: float=1.0):
        x_hat, mean, log_var = self.forward(x)
        # kl_div = -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp())
        kl_div = torch.mean(-0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var),dim=1), dim=0)
        reconstruction_loss = nn.functional.mse_loss(x, x_hat)
        loss = reconstruction_loss + beta * kl_div
        return loss, reconstruction_loss, kl_div

    def sample(self, n_samples: int):
        z = torch.randn(n_samples, self.latent_dim)
        return self.decode(z)

    def forward(self, x: torch.Tensor):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var


#%%

# TODO: Combine with StoreState, and make so that u_int_batch gets handled in a clever way -- i.e.
#      for training, keep the same one, but then for eval, use a new one (starting as zeros).
class SolutionOfPDENet(nn.Module):
    def __init__(self,
                 get_N: Callable[[], Callable[[torch.Tensor], torch.Tensor]],
                 x: torch.Tensor,
                 latent_dim: int,
                 stab_term: float = 0.0,
                 newton_kwaargs: Optional[dict] = None,
                 init_kwaargs: Optional[dict] = {'max_iter': 100, 'n_batch': 10}) -> None:
        """
        Batched (with a loop) operation of the solution `N(u, u_x, u_{xx} ; z) = 0`, i.e. batched `S o N`

        Args:
            N: Dynamics giving residual `N(u, u_x, u_{xx} ; z) = 0`, `[n_x, 4+latent_dim]->[n_x, 1]`.
            x: Grid points for the PDE solution. `[n_x, ]`.
            newton_kwaargs: Dictionary of keyword arguments for the newton solver.
        """
        super(SolutionOfPDENet, self).__init__()

        self.x = x
        self.latent_dim = latent_dim
        self.stab_term = stab_term
        self.newton_kwaargs = {'max_iter': 20, 'damp': 1e0, 'verbose': False} \
            if newton_kwaargs is None else newton_kwaargs
        self.get_N = get_N
        self._init_N(**init_kwaargs)

    def _init_N(self, max_iter, n_batch):
        success = False
        for i in range(max_iter):
            self.N = self.get_N()
            if self._is_good_N(n_batch):
                success = True
                break

            if max_iter < 10 or i % (max_iter // 10) == 0:
                print(f'Failed to find a good N initialization after {i+1}/{max_iter} iterations. Trying again.')

        if not success:
            raise ValueError(f'Could not find a good N after {max_iter} iterations.')

    def _is_good_N(self, n_batch):
        # Check for newton's iterations, or try catch for gradient of arbitrary loss
        z = torch.randn(n_batch, self.latent_dim)

        u_solve_batch = torch.zeros(n_batch, len(self.x)-2)
        with torch.no_grad():
            u_solve_batch = self.forward(u_solve_batch, z)

        # check if any are nan
        if torch.isnan(u_solve_batch).any():
            return False

        # now do it again, but with the gradients
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0)
        for i in range(2):
            optimizer.zero_grad()
            u_solve_batch = self.forward(u_solve_batch, z)
            loss = torch.mean(u_solve_batch**2)
            loss.backward()
            optimizer.step()
            u_solve_batch = u_solve_batch.detach()
            if loss.isnan():
                return False

        return True


    def _general_f(self, u_int, z):
        """"
        Repeats `z` spatially and computes the residual `N(u, u_x, u_{xx} ; z)` at each grid point.
        """
        u = torch.zeros_like(self.x)
        u[1:-1] = u_int
        ux = utils.fd_centered(u, self.x)
        uxx = utils.fd_centered_2nd(u, self.x)
        big_u = torch.stack([self.x[1:-1], u[1:-1], ux, uxx], dim=1)
        big_u_z = torch.cat([big_u, z.repeat(big_u.shape[0], 1)], dim=1)
        N_res = self.N(big_u_z)[:,0]
        stab_N_res = N_res + self.stab_term *  uxx
        return stab_N_res

    def _single_forward(self, u_int, z):
        """
        Newton solve of `_general_f` for a single `u_int` and `z`.
        """
        f = lambda u_int: self._general_f(u_int, z)
        u_int = utils.newton_solve(f, u_int, **self.newton_kwaargs)
        return u_int

    def forward(self, u_int_batch, z_batch):
        """
        Batched solution operation of `N(u, u_x, u_{xx} ; z) = 0`

        Args:
            u_int_batch: Batch of initial guesses for the solution. `[n_batch, n_x-2]`
            z_batch: Batch of latent variables. `[n_batch, latent_dim]`. Does not vary spatially.
        Returns:
            u_int_batch: Batch of solved solutions. `[n_batch, n_x-2]`
        """

        for i in range(u_int_batch.shape[0]):
            u_int = u_int_batch[i]
            z = z_batch[i]
            u_int = self._single_forward(u_int, z)
            u_int_batch[i] = u_int

        return u_int_batch


#%% SIREN
# Taken from
# https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=JMOfAQiuA0_J
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

#%% OLD: BAD SIREN
class SimpleFNN(nn.Module):
    def __init__(self, layers, activation):
        super(SimpleFNN, self).__init__()
        self.layers = layers
        self.activation = activation
        self.fc_layers = nn.ModuleList()

        # Create fully connected layers
        for i in range(len(layers) - 1):
            self.fc_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.fc_layers.append(activation)

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x

class SineActivation(nn.Module):
    def __init__(self):
        super(SineActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)
