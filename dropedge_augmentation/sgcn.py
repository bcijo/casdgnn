import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # Add this missing import
from torch_geometric.nn import MessagePassing
from load_dataset import load_bitcoin_dataset

class SGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SGCNConv, self).__init__(aggr='add')
        self.pos_weight = nn.Parameter(torch.Tensor(in_channels * 2, out_channels))
        self.neg_weight = nn.Parameter(torch.Tensor(in_channels * 2, out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.pos_weight)
        nn.init.xavier_uniform_(self.neg_weight)
    
    def forward(self, x, pos_edge_index, neg_edge_index):
        # Positive propagation
        pos_x = self.propagate(pos_edge_index, x=x)
        pos_out = torch.cat([pos_x, x], dim=-1)
        pos_out = F.relu(pos_out @ self.pos_weight)
        
        # Negative propagation
        neg_x = self.propagate(neg_edge_index, x=x)
        neg_out = torch.cat([neg_x, x], dim=-1)
        neg_out = F.relu(neg_out @ self.neg_weight)
        
        return pos_out, neg_out

class SGCN(nn.Module):
    def __init__(self, n_nodes, hidden_dim=64, num_layers=2):
        super(SGCN, self).__init__()
        self.emb = nn.Parameter(torch.randn(n_nodes, hidden_dim))
        self.layers = nn.ModuleList()
        
        current_in_dim = hidden_dim
        for i in range(num_layers):
            # Each SGCNConv layer outputs hidden_dim for pos_path and hidden_dim for neg_path
            self.layers.append(SGCNConv(current_in_dim, hidden_dim))
            # Input to the next layer will be the concatenation of pos and neg paths
            current_in_dim = hidden_dim * 2 
            
        # The final x from SGCN.forward is a concatenation of pos_x and neg_x from the last layer,
        # so its dimension is hidden_dim * 2.
        # In predict_edge, we concatenate two such node embeddings (z_i, z_j),
        # so the input to mlg becomes (hidden_dim * 2) * 2 = hidden_dim * 4.
        self.mlg = nn.Linear(hidden_dim * 4, 3)  # Adjusted input dimension
    
    def forward(self, pos_edge_index, neg_edge_index):
        x = self.emb
        for layer in self.layers:
            pos_x, neg_x = layer(x, pos_edge_index, neg_edge_index)
            x = torch.cat([pos_x, neg_x], dim=-1)
        return x
    
    def predict_edge(self, x, edge_index):
        z_i = x[edge_index[0]]
        z_j = x[edge_index[1]]
        z = torch.cat([z_i, z_j], dim=-1)
        return self.mlg(z)

# Training function
def train_sgcn(model, edge_index, edge_signs, pos_edge_index, neg_edge_index, epochs=300, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # pos_edge_index and neg_edge_index for the full graph are used here for message passing
        z = model(pos_edge_index, neg_edge_index) 
        logits = model.predict_edge(z, edge_index) # edge_index here is train_edge_index
        loss = F.cross_entropy(logits, (edge_signs + 1).long())  # Convert signs to 0,1,2
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    return model

def evaluate_sgcn(model, edge_index, edge_signs, pos_edge_index, neg_edge_index): # Added pos/neg_edge_index
    model.eval()
    with torch.no_grad():
        # pos_edge_index and neg_edge_index for the full graph are used here for message passing
        z = model(pos_edge_index, neg_edge_index)
        logits = model.predict_edge(z, edge_index) # edge_index here is test_edge_index
        preds = logits.argmax(dim=1)
        true_labels = (edge_signs + 1).long()  # Convert signs to 0,1,2
        
        accuracy = (preds == true_labels).float().mean().item()
        
        # Per-class accuracy
        pos_acc = (preds[true_labels == 2] == 2).float().mean().item() if (true_labels == 2).any() else 0
        neg_acc = (preds[true_labels == 0] == 0).float().mean().item() if (true_labels == 0).any() else 0
        
        return {
            'accuracy': accuracy,
            'positive_accuracy': pos_acc,
            'negative_accuracy': neg_acc
        }

# Example usage
print("Loading Bitcoin dataset...")

file_path = 'bitcoin_alpha.csv'  # Make sure this path is correct
# Unpack the newly returned pos_edge_index and neg_edge_index
G, node2idx, idx2node, pos_adj, neg_adj, edge_index, edge_signs, pos_edge_index, neg_edge_index = load_bitcoin_dataset(file_path)

print("Initializing SGCN model...")
n_nodes = len(G.nodes())
sgcn = SGCN(n_nodes)

# Split edges into train/test
num_edges = edge_index.size(1)
num_test = int(0.2 * num_edges)

# Create a random permutation of indices
perm = torch.randperm(num_edges)
test_indices = perm[:num_test]
train_indices = perm[num_test:]

# Create masks from the random indices
test_mask = torch.zeros(num_edges, dtype=torch.bool)
test_mask[test_indices] = 1
train_mask = ~test_mask # Or train_mask[train_indices] = 1, but ~test_mask is more direct

train_edge_index = edge_index[:, train_mask]
train_edge_signs = edge_signs[train_mask]
test_edge_index = edge_index[:, test_mask]
test_edge_signs = edge_signs[test_mask]

num_train_edges = train_edge_index.size(1)
num_test_edges = test_edge_index.size(1)

# Calculate edge type statistics
train_pos = (train_edge_signs == 1).sum().item()
train_neg = (train_edge_signs == -1).sum().item()
test_pos = (test_edge_signs == 1).sum().item()
test_neg = (test_edge_signs == -1).sum().item()

# Print detailed split statistics
print(f"\n--- Data Split Statistics (Random Split) ---")
print(f"Total edges: {num_edges}")
print(f"Training edges: {num_train_edges} ({100 * num_train_edges / num_edges:.2f}%)")
print(f"  - Positive: {train_pos} ({100 * train_pos / num_train_edges:.2f}%)")
print(f"  - Negative: {train_neg} ({100 * train_neg / num_train_edges:.2f}%)")
print(f"Testing edges: {num_test_edges} ({100 * num_test_edges / num_edges:.2f}%)")
print(f"  - Positive: {test_pos} ({100 * test_pos / num_test_edges:.2f}%)")
print(f"  - Negative: {test_neg} ({100 * test_neg / num_test_edges:.2f}%)")

print(f"\nStarting training on {num_train_edges} edges, testing on {num_test_edges} edges...")

# Train model
# Pass the full graph's pos_edge_index and neg_edge_index for message passing context
sgcn = train_sgcn(sgcn, train_edge_index, train_edge_signs, pos_edge_index, neg_edge_index)

# Evaluate
print("\nEvaluating model...")
# Pass the full graph's pos_edge_index and neg_edge_index for message passing context
metrics = evaluate_sgcn(sgcn, test_edge_index, test_edge_signs, pos_edge_index, neg_edge_index)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Positive Edge Accuracy: {metrics['positive_accuracy']:.4f}")
print(f"Negative Edge Accuracy: {metrics['negative_accuracy']:.4f}")

# Verify some predictions
print("\nSample predictions:")
with torch.no_grad():
    # Pass the full graph's pos_edge_index and neg_edge_index for message passing context
    z = sgcn(pos_edge_index, neg_edge_index)
    sample_edges = test_edge_index[:, :5]  # Look at first 5 test edges
    sample_preds = sgcn.predict_edge(z, sample_edges).argmax(dim=1)
    sample_true = (test_edge_signs[:5] + 1).long()
    
    for i in range(5):
        print(f"Edge {sample_edges[:,i].tolist()}: Predicted {sample_preds[i]-1}, True {sample_true[i]-1}")