import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import AffineCouplingTransform, RandomPermutation
from nflows.transforms.base import CompositeTransform
from nflows.nn.nets import ResidualNet


class ConditionalNormalizingFlowModel(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dim, num_layers, device):
        super(ConditionalNormalizingFlowModel, self).__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.device = device  # Store the device

        # Base distribution for the 4D subspace (input_dim = 4)
        self.base_distribution = StandardNormal(shape=[input_dim])

        # Create a sequence of Affine Coupling Transforms, handling conditioning on the 5th dimension via the transform network
        transforms = []
        for i in range(num_layers):
            # Alternating mask for coupling layers (splitting the dimensions)
            mask = torch.tensor([i % 2] * (input_dim // 2) + [(i + 1) % 2] * (input_dim - input_dim // 2))  # Keep mask on CPU

            # Define a transform with a custom neural network, using context as an additional input in the transform network
            transforms.append(AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,   # No context concatenation here
                    out_features=out_features,
                    hidden_features=hidden_dim,
                    context_features=context_dim,  # This allows us to condition on the context
                    num_blocks=2
                ).to(self.device)
            ))
            transforms.append(RandomPermutation(features=input_dim))  # Randomly permute after each layer

        self.transform = CompositeTransform(transforms).to(self.device)
        self.flow = Flow(self.transform, self.base_distribution).to(self.device)

    def forward(self, x, context):
        # Move x and context to the correct device
        x = x.to(self.device)
        context = context.to(self.device)
        return self.flow.log_prob(x, context)

    def sample(self, num_samples, context):
        # Ensure context is on the correct device
        context = context.to(self.device)
        return self.flow.sample(num_samples, context)

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dim, num_timesteps, device):
        super(ConditionalDiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.num_timesteps = num_timesteps
        self.device = device  # Store the device

        # Define a neural network to predict noise at each timestep conditioned on input and context
        self.denoise_net = nn.Sequential(
            nn.Linear(input_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        ).to(self.device)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffusion forward process: Adds noise to data based on timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)

        alpha_t = self.get_alpha(t).to(self.device)
        return alpha_t * x_start + (1 - alpha_t) * noise

    def get_alpha(self, t):
        """
        Returns the alpha coefficient for a given timestep t (shape: [batch_size, 1]).
        """
        beta_t = torch.linspace(0.0001, 0.02, self.num_timesteps).to(self.device)
        alpha_t = 1 - beta_t[t]
        return alpha_t.view(-1, 1)

    def p_sample(self, x_t, t, context):
        """
        Reverse process (sampling): Denoises the input at time t conditioned on the context.
        """
        # Concatenate input and context
        x_t_context = torch.cat([x_t, context], dim=-1).to(self.device)

        # Predict noise using the denoise network
        predicted_noise = self.denoise_net(x_t_context)

        alpha_t = self.get_alpha(t).to(self.device)
        return (x_t - (1 - alpha_t) * predicted_noise) / alpha_t

    def forward(self, x_start, context):
        """
        Forward pass of the diffusion model: Perform the forward diffusion process and then reverse it.
        """
        batch_size = x_start.size(0)
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()

        # Forward diffusion: Adding noise
        noise = torch.randn_like(x_start).to(self.device)
        x_noisy = self.q_sample(x_start, timesteps, noise)

        # Reverse process: Denoise using the context
        x_reconstructed = self.p_sample(x_noisy, timesteps, context)

        # Return the predicted noise and actual noise to compute the loss
        return x_reconstructed, noise

    def sample(self, num_samples, context):
        """
        Sample new data by reversing the diffusion process.
        """
        x_t = torch.randn(num_samples, self.input_dim).to(self.device)
        for t in reversed(range(self.num_timesteps)):
            x_t = self.p_sample(x_t, t, context)
        return x_t
