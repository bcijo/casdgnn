import torch
import torch.nn.functional as F
import os
import sys

# Add the parent directory to the path to import from dropedge_augmentation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SGCN components
from sgcn import SGCN, train_sgcn, evaluate_sgcn
from load_dataset import load_bitcoin_dataset

# Example usage and testing
print("=" * 60)
print("TESTING SGCN MODEL")
print("=" * 60)

print("Loading Bitcoin dataset...")

file_path = r'D:\abhin\Comding\ML\Capstone\casdgnn\dropedge_augmentation\bitcoin_alpha.csv' 
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
print("Training SGCN model...")
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

print("\n" + "=" * 60)
print("SGCN MODEL TESTING COMPLETED")
print("=" * 60)