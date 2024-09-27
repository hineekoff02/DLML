import glob
import tqdm
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import AffineCouplingTransform, RandomPermutation
from nflows.transforms.base import CompositeTransform
from nflows.nn.nets import ResidualNet

# Define the normalizing flow model
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


# Training the flow model
def train_conditional_flow_model(flow_model, data, context, num_epochs=1000, batch_size=512, learning_rate=1e-3):
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=learning_rate)

    # Convert data and context to tensors and move them to the same device as the model
    data = torch.tensor(data, dtype=torch.float32, device=flow_model.device)
    context = torch.tensor(context, dtype=torch.float32, device=flow_model.device)
    
    dataset = torch.utils.data.TensorDataset(data, context)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    all_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm.tqdm(dataloader, desc=f"Training epoch {epoch}"):
            batch_data, batch_context = batch
            optimizer.zero_grad()
            loss = -flow_model(batch_data, batch_context).mean()  # Maximize the log probability
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Print loss every epoch
        print(f"Epoch {epoch}, Loss: {total_loss}")
        all_losses.append(total_loss / len(dataloader.dataset))

    # Save loss data to a CSV file
    df = pd.DataFrame({"loss": all_losses})
    df.to_csv("loss.csv")


# Function to generate samples conditioned on the 5th dimension
def generate_samples(flow_model, num_samples, fixed_value_5th_dim):
    # The context is the fixed value for the 5th dimension, we expand it to match num_samples
    context = torch.tensor(fixed_value_5th_dim, device=flow_model.device).repeat(num_samples, 1)
    
    # Sample from the learned distribution conditioned on the 5th dimension
    samples = flow_model.sample(num_samples, context)
    
    return samples


def concat_files(filelist):
    all_data = None
    for f in tqdm.tqdm(filelist, desc="Loading data into array"):
        if all_data is None:
            all_data = np.load(f)[:, :4]
        else:
            all_data = np.concatenate((all_data, np.load(f)[:, :4]))
    energy = np.sum(all_data, axis=1).reshape(-1, 1)
    
    return np.concatenate((all_data, energy), axis=1)


# Example usage
if __name__ == "__main__":
    # Loading data
    files_train = glob.glob("/ceph/bmaier/delight/ml/nf/data/train/*npy")
    random.seed(123)
    random.shuffle(files_train)
    data = concat_files(files_train)
    
    print(data.shape)
    
    # Separate the data into the first 4 dimensions (input) and the 5th dimension (context)
    data_4d = data[:, :4]
    context_5d = data[:, 4:5]

    # Initialize the conditional flow model (input dimension 4, context dimension 1, hidden dimension 64, 5 layers)
    input_dim = 4
    context_dim = 1
    hidden_dim = 64
    num_layers = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Create and move the model to the appropriate device
    flow_model = ConditionalNormalizingFlowModel(input_dim, context_dim, hidden_dim, num_layers, device).to(device)
    
    # Train the model
    train_conditional_flow_model(flow_model, data_4d, context_5d, num_epochs=200)

    # Optionally, generate samples conditioned on a specific value of the 5th dimension
    fixed_value_5th_dim = torch.tensor([[0.5]])  # For example, conditioning on 5th dim being 0.5
    generated_samples = generate_samples(flow_model, num_samples=1000, fixed_value_5th_dim=fixed_value_5th_dim)

    print("Generated 4D samples conditioned on 5th dimension value:\n", generated_samples)
