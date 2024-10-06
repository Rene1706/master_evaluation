import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import ast
from pathlib import Path
pruning = False

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

def densify_and_prune(opacities, min_opacity):
    # Prune points based on opacity (prune if opacity is less than min_opacity)
    prune_mask = opacities.squeeze() < min_opacity
    return prune_mask


def generate_action_labels(grads, scaling, opacities, grad_threshold, percent_dense, scene_extent, min_opacity):
    # Get the maximum scaling value (even though scaling is 1D now, we keep this logic in case it's expanded)
    max_scalings = scaling

    # Create action labels: 0 = do nothing, 1 = clone, 2 = split, 3 = prune
    clone_mask = densify_and_clone(grads, grad_threshold, max_scalings, percent_dense, scene_extent)
    split_mask = densify_and_split(grads, grad_threshold, max_scalings, percent_dense, scene_extent)
    if pruning:
        prune_mask = densify_and_prune(opacities, min_opacity)

    action_labels = torch.zeros(grads.size(0), dtype=torch.long, device=grads.device)
    action_labels[split_mask] = 2  # Split
    action_labels[clone_mask] = 1  # Clone
    if pruning:
        action_labels[prune_mask] = 3  # Prune
    
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
    def __init__(self, input_size, hidden_size=16, output_size=4):
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
            # ! Why do i not need softmax to make a real decision
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
    file_path = '/bigwork/nhmlhuer/gaussian-splatting/grad_and_scaling_and_opacity_2.txt'

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
            opacity_stats.append({'mean': entry['mean_opacity'], 'std': entry['std_opacity']})

    return grad_stats, scaling_stats, opacity_stats

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Number of samples per phase (start, mid, late)
    num_samples_per_range = 50000
    
    grad_stats, scaling_stats, opacity_stats = get_stats()
    #print(grad_stats)
    # Empty lists to collect inputs and action labels
    all_inputs = []
    all_actions = []

    # Set constants for the thresholds
    grad_threshold = 0.0002
    percent_dense = 0.01
    scene_extent = 4.802176904678345
    min_opacity = 0.005  # Opacity threshold for pruning

    # Generate samples and action labels for each phase
    for grad_stat, scaling_stat, opacity_stat in zip(grad_stats, scaling_stats, opacity_stats):
        # Generate grads, scaling, and opacities for the current phase
        grads = torch.normal(mean=grad_stat['mean'], std=grad_stat['std'], size=(num_samples_per_range,), device=device)
        scaling = torch.normal(mean=scaling_stat['mean'], std=scaling_stat['std'], size=(num_samples_per_range,), device=device)
        opacities = torch.normal(mean=opacity_stat['mean'], std=opacity_stat['std'], size=(num_samples_per_range,), device=device)
        #print("Opactiies: ", opacities)
        #print(grads)
        #print(grads.min(), grads.max())
        # Generate action labels using the densification functions, including pruning
        actions = generate_action_labels(grads, scaling, opacities, grad_threshold, percent_dense, scene_extent, min_opacity)
        #print("Actions: ", actions)
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
    output_size = 4 if pruning else 3
    param_network = ParamNetwork(input_size=3, output_size=output_size).to(device)
    model_name = "imitation_learning_no_pruning_model_long.torch"
    if Path(model_name).exists():
        print("Loading pre-trained model...")
        param_network.load_state_dict(torch.load(model_name))
    # Train the imitation model
    epochs = 15
    batch_size = 256
    learning_rate = 1e-3
    train_imitation_model(param_network, dataset, epochs=epochs, batch_size=batch_size, lr=learning_rate)

    # Save the trained model's state dict with the specified filename
    torch.save(param_network.state_dict(), model_name)
    print(f"Training completed and model saved to {model_name}.")

if __name__ == "__main__":
    main()
