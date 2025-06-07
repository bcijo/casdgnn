import networkx as nx
import torch
import torch.nn.functional as F
from itertools import combinations
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the functions from edge_difficulty.py
from edge_difficulty import compute_edge_difficulty, curriculum_training
# Import the is_balanced_triangle function from select_candidates.py
from select_candidates import is_balanced_triangle

# Mock SGCN model for testing purposes
class MockSGCN:
    def __init__(self, node_dim=64, num_nodes=10):
        self.node_dim = node_dim
        self.num_nodes = num_nodes
        # Simple linear layers for demonstration
        self.conv1 = torch.nn.Linear(node_dim, 32)
        self.conv2 = torch.nn.Linear(32, 16)
        self.predictor = torch.nn.Linear(32, 2)  # Binary classification for edge signs
        
    def parameters(self):
        # Return all model parameters for optimizer
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.predictor.parameters())
        return params
    
    def train(self):
        # Set model to training mode
        self.conv1.train()
        self.conv2.train()
        self.predictor.train()
    
    def __call__(self, pos_edge_index, neg_edge_index):
        # Generate mock node embeddings
        # In reality, this would use the graph structure
        return torch.randn(self.num_nodes, 16)
    
    def predict_edge(self, z, edge_index):
        # Simple edge prediction based on node embeddings
        # z: node embeddings [num_nodes, embedding_dim]
        # edge_index: [2, num_edges]
        
        src_embeddings = z[edge_index[0]]  # Source node embeddings
        dst_embeddings = z[edge_index[1]]  # Destination node embeddings
        
        # Concatenate source and destination embeddings
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Predict edge signs
        logits = self.predictor(edge_embeddings)
        return logits

# Example Usage and Testing

print("=" * 60)
print("TESTING EDGE DIFFICULTY COMPUTATION")
print("=" * 60)

# 1. Create a sample augmented graph G_aug
G_aug = nx.Graph()
G_aug.add_edge(0, 1, sign=1)
G_aug.add_edge(1, 2, sign=1)
G_aug.add_edge(2, 0, sign=1)  # Balanced triangle (0,1,2)
G_aug.add_edge(2, 3, sign=-1)
G_aug.add_edge(3, 4, sign=1)
G_aug.add_edge(0, 4, sign=1)  # From previous augmentation

print("Graph G_aug edges:")
for u, v, data in G_aug.edges(data=True):
    print(f"  ({u}, {v}, sign={data['sign']})")
print()

# 2. Create edge_index tensor (PyTorch format)
edges = list(G_aug.edges())
edge_index = torch.tensor([[u, v] for u, v in edges]).t().contiguous()
print("Edge index tensor:")
print(edge_index)
print()

# 3. Test compute_edge_difficulty function
print("Computing edge difficulty scores...")
try:
    difficulty_scores = compute_edge_difficulty(G_aug, edge_index)
    
    print("Difficulty scores for each edge:")
    for edge, score in difficulty_scores.items():
        print(f"  Edge {edge}: {score:.4f}")
    print()
    
except Exception as e:
    print(f"Error in compute_edge_difficulty: {e}")
    print("This might be due to missing imports in edge_difficulty.py")
    print()

print("=" * 60)
print("TESTING CURRICULUM TRAINING")
print("=" * 60)

# 4. Set up data for curriculum training
num_nodes = len(G_aug.nodes())
num_edges = len(edges)

# Create mock SGCN model
sgcn = MockSGCN(node_dim=64, num_nodes=num_nodes)

# Create edge signs tensor
edge_signs = torch.tensor([G_aug[u][v]['sign'] for u, v in edges], dtype=torch.float)
print(f"Edge signs: {edge_signs}")
print(f"Unique edge signs: {torch.unique(edge_signs)}")

# Create pos_edge_index and neg_edge_index
pos_edges = [(u, v) for u, v in edges if G_aug[u][v]['sign'] == 1]
neg_edges = [(u, v) for u, v in edges if G_aug[u][v]['sign'] == -1]

if pos_edges:
    pos_edge_index = torch.tensor(pos_edges).t().contiguous()
else:
    pos_edge_index = torch.empty((2, 0), dtype=torch.long)

if neg_edges:
    neg_edge_index = torch.tensor(neg_edges).t().contiguous()
else:
    neg_edge_index = torch.empty((2, 0), dtype=torch.long)

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(f"Positive edges: {len(pos_edges)}")
print(f"Negative edges: {len(neg_edges)}")
print()

# 5. Test curriculum training (with reduced epochs for testing)
print("Starting curriculum training (reduced epochs for testing)...")
try:
    # Use the difficulty scores computed earlier or create mock ones
    if 'difficulty_scores' not in locals():
        # Create mock difficulty scores
        difficulty_scores = {(u, v): np.random.random() for u, v in edges}
        print("Using mock difficulty scores since compute_edge_difficulty failed")
    
    # Run curriculum training with reduced parameters
    trained_model = curriculum_training(
        model=sgcn,
        edge_index=edge_index,
        edge_signs=edge_signs,
        difficulty_scores=difficulty_scores,
        pos_edge_index=pos_edge_index,
        neg_edge_index=neg_edge_index,
        epochs=50,  # Reduced for testing
        T=20,       # Reduced for testing
        lambda_0=0.2
    )
    
    print("Curriculum training completed successfully!")
    print()
    
except Exception as e:
    print(f"Error in curriculum_training: {e}")
    print("This might be due to missing imports or model implementation issues")
    print()

print("=" * 60)
print("TESTING EDGE DIFFICULTY ANALYSIS")
print("=" * 60)

# 6. Analyze the difficulty scores
if 'difficulty_scores' in locals() and difficulty_scores:
    print("Analysis of difficulty scores:")
    scores = list(difficulty_scores.values())
    print(f"  Min difficulty: {min(scores):.4f}")
    print(f"  Max difficulty: {max(scores):.4f}")
    print(f"  Mean difficulty: {np.mean(scores):.4f}")
    print(f"  Std difficulty: {np.std(scores):.4f}")
    print()
    
    # Sort edges by difficulty (easiest to hardest)
    sorted_edges = sorted(difficulty_scores.items(), key=lambda x: x[1])
    print("Edges sorted by difficulty (easiest to hardest):")
    for i, (edge, score) in enumerate(sorted_edges):
        print(f"  {i+1}. Edge {edge}: {score:.4f}")
    print()

print("=" * 60)
print("TESTING PACING FUNCTION")
print("=" * 60)

# 7. Test the pacing function separately
print("Testing pacing function g(t) = min(1, λ₀ + (1-λ₀) * t/T):")
lambda_0 = 0.2
T = 100
total_edges = len(edges)

sample_epochs = [0, 10, 25, 50, 75, 100, 150, 200, 300]
print(f"λ₀ = {lambda_0}, T = {T}, Total edges = {total_edges}")
print()
print("Epoch | g(t)   | Edges used | Percentage")
print("------|--------|------------|----------")

for epoch in sample_epochs:
    g_t = min(1, lambda_0 + (1 - lambda_0) * epoch / T)
    num_edges_used = int(total_edges * g_t)
    percentage = (num_edges_used / total_edges) * 100
    print(f"{epoch:5d} | {g_t:.3f} | {num_edges_used:10d} | {percentage:7.1f}%")

print()
print("Test completed!")
print("=" * 60)