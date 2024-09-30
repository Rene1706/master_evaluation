import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import ast
from pathlib import Path

def densify_and_clone(grads, grad_threshold, scaling, percent_dense, scene_extent):
    # Extract points that satisfy the gradient condition for cloning
    #print("Grads:", grads)
    # Ensure grads is 2D
    if grads.dim() == 1:
        grads = grads.unsqueeze(-1)
    #print("Grads: ", grads)
    #print("norm: ", torch.norm(grads, dim=-1))
    selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
    selected_pts_mask = torch.logical_and(
        selected_pts_mask, scaling <= percent_dense * scene_extent
    )
    #print("Selected_pts_mask: ", selected_pts_mask)
    return selected_pts_mask

def densify_and_split(grads, grad_threshold, scaling, percent_dense, scene_extent, N=2):
    # Extract points that satisfy the gradient condition for splitting
    selected_pts_mask = grads.squeeze() >= grad_threshold
    selected_pts_mask = torch.logical_and(
        selected_pts_mask, scaling > percent_dense * scene_extent
    )
    #print("Scaliog: ", scaling)
    #print("Mul; ", percent_dense * scene_extent)
    return selected_pts_mask


def generate_sample_data_multiple_ranges_with_opacities(num_samples_per_range, device='cuda'):
    # Defining multiple mean and standard deviation values for grad, scaling, and opacity
    # These represent start, mid, and late phases

    grad_stats = [
        #{'mean': 7.207840098999441e-06, 'std': 3.625786121119745e-05},  # start
        #{'mean': 3.781244231504388e-05, 'std': 0.00010330345685360953},  # mid
        #{'mean': 2.1055266188341193e-05, 'std': 6.029089854564518e-05},  # late
        #{'mean': 2e-4, 'std': 5e-6},  # start
        #{'mean': 2e-4, 'std': 5e-5},  # start
        #{'mean': 2e-4, 'std': 5e-4},  # start
        #{'mean': 2e-8, 'std': 5e-10},  # start
    ]

    scaling_stats = [
        #{'mean': 0.0133, 'std': 0.0040},  # start
        #{'mean': 0.0179, 'std': 0.0072},  # mid
        #{'mean': 0.0198, 'std': 0.0126},  # late
        #{'mean': 0.048, 'std': 0.01},  # start
        #{'mean': 0.048, 'std': 0.01},  # start
        #{'mean': 0.048, 'std': 0.01},  # start
        #{'mean': 0.048, 'std': 0.01},  # start
    ]
    
    # Assuming similar ranges for opacities
    opacity_stats = [
        {'mean': 0.007, 'std': 0.001},  # start
        {'mean': 0.007, 'std': 0.001},  # start
        {'mean': 0.007, 'std': 0.001},  # start
        #{'mean': 0.007, 'std': 0.001},  # start
    ]

    all_grads = []
    all_scalings = []
    all_opacities = []

    # Sample for each range (start, mid, late)
    for grad_stat, scaling_stat, opacity_stat in zip(grad_stats, scaling_stats, opacity_stats):
        #print(grad_stat, scaling_stat, opacity_stat)
        grads = torch.normal(mean=grad_stat['mean'], std=grad_stat['std'], size=(num_samples_per_range,), device=device)
        scaling = torch.normal(mean=scaling_stat['mean'], std=scaling_stat['std'], size=(num_samples_per_range,), device=device)
        opacities = torch.normal(mean=opacity_stat['mean'], std=opacity_stat['std'], size=(num_samples_per_range,), device=device)
        #print(grads.min(), grads.max())
        #print("grads: ", grads)
        all_grads.append(grads)
        all_scalings.append(scaling)
        all_opacities.append(opacities)

    # Concatenate all the samples from different phases
    all_grads = torch.cat(all_grads, dim=0)
    all_scalings = torch.cat(all_scalings, dim=0)
    all_opacities = torch.cat(all_opacities, dim=0)

    # Set thresholds and other constants
    grad_threshold = 0.0002
    percent_dense = 0.01
    scene_extent = 4.802176904678345
    #print("All_grads: ", all_grads)

    return all_grads, all_scalings, all_opacities, grad_threshold, percent_dense, scene_extent



def generate_action_labels(grads, scaling, opacities, grad_threshold, percent_dense, scene_extent):
    # Get the maximum scaling value (even though scaling is 1D now, we keep this logic in case it's expanded)
    max_scalings = scaling

    # Create action labels: 0 = do nothing, 1 = split, 2 = clone
    clone_mask = densify_and_clone(grads, grad_threshold, max_scalings, percent_dense, scene_extent)
    split_mask = densify_and_split(grads, grad_threshold, max_scalings, percent_dense, scene_extent)
    #print(clone_mask)
    #print(split_mask)

    action_labels = torch.zeros(grads.size(0), dtype=torch.long, device=grads.device)
    action_labels[split_mask] = 2  # Split
    action_labels[clone_mask] = 1  # Clone
    #print("Length of action labels: ", len(action_labels))
    #print(action_labels)
    return action_labels


class ImitationDataset(Dataset):
    def __init__(self, inputs, actions):
        self.inputs = inputs
        self.actions = actions

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.actions[idx]


class ParamNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=16, output_size=3):
        super(ParamNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_imitation_model(param_network, dataset, epochs=10, batch_size=64, lr=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(param_network.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_inputs, batch_actions in dataloader:
            optimizer.zero_grad()
            outputs = param_network(batch_inputs)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_inputs.size(0)

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += batch_actions.size(0)
            correct += (predicted == batch_actions).sum().item()

        avg_loss = epoch_loss / len(dataset)
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

def get_stats():
    # Path to the text file
    file_path = '/bigwork/nhmlhuer/gaussian-splatting/grad_and_scaling_4.txt'

    # Lists to store the parsed values
    grad_stats = []
    scaling_stats = []
    opacity_stats = []

    # Read the file and parse each entry
    with open(file_path, 'r') as f:
        for line in f:
            # Parse the line as a dictionary
            entry = ast.literal_eval(line.strip())
            
            # Extract grad and scaling stats
            grad_stats.append({'mean': entry['mean_grad_value'], 'std': entry['std_grad_value']})
            scaling_stats.append({'mean': entry['mean_scaling'], 'std': entry['std_scaling']})
            
            # Assuming default opacity values
            opacity_stats.append({'mean': 0.7, 'std': 0.1})

    return grad_stats, scaling_stats, opacity_stats

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Number of samples per phase (start, mid, late)
    num_samples_per_range = 10000
    
    grad_stats, scaling_stats, opacity_stats = get_stats()
    #print(grad_stats)
    # Empty lists to collect inputs and action labels
    all_inputs = []
    all_actions = []

    # Set constants for the thresholds
    grad_threshold = 0.0002
    percent_dense = 0.01
    scene_extent = 4.802176904678345

    # Generate samples and action labels for each phase
    for grad_stat, scaling_stat, opacity_stat in zip(grad_stats, scaling_stats, opacity_stats):
        # Generate grads, scaling, and opacities for the current phase
        grads = torch.normal(mean=grad_stat['mean'], std=grad_stat['std'], size=(num_samples_per_range,), device=device)
        scaling = torch.normal(mean=scaling_stat['mean'], std=scaling_stat['std'], size=(num_samples_per_range,), device=device)
        opacities = torch.normal(mean=opacity_stat['mean'], std=opacity_stat['std'], size=(num_samples_per_range,), device=device)
        #print(grads)
        #print(grads.min(), grads.max())
        # Generate action labels using the densification functions
        actions = generate_action_labels(grads, scaling, opacities, grad_threshold, percent_dense, scene_extent)

        # Concatenate inputs: grads, scaling, and opacities for this phase
        inputs = torch.cat([grads.unsqueeze(-1), scaling.unsqueeze(-1), opacities.unsqueeze(-1)], dim=-1)

        # Append to the overall list of inputs and actions
        all_inputs.append(inputs)
        all_actions.append(actions)

    # Concatenate inputs and actions across all phases
    all_inputs = torch.cat(all_inputs, dim=0)
    all_actions = torch.cat(all_actions, dim=0)

    # Create dataset with inputs (gradients, scaling, opacities) and actions
    dataset = ImitationDataset(all_inputs, all_actions)

    # Initialize the RL agent (ParamNetwork)
    input_size = all_inputs.shape[1]  # Should be 3
    param_network = ParamNetwork(input_size=3).to(device)
    if Path("imitation_learning_model.torch").exists():
        param_network.load_state_dict(torch.load("imitation_learning_model.torch"))
    # Train the imitation model
    epochs = 10
    batch_size = 16
    learning_rate = 1e-3
    train_imitation_model(param_network, dataset, epochs=epochs, batch_size=batch_size, lr=learning_rate)

    # Save the trained model's state dict with the specified filename
    model_save_path = "imitation_learning_model.torch"
    torch.save(param_network.state_dict(), model_save_path)
    print(f"Training completed and model saved to {model_save_path}.")

if __name__ == "__main__":
    main()
