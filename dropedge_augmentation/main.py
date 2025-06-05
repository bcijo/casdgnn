import torch
from load_dataset import load_bitcoin_dataset
from sgcn import SGCN, train_sgcn
from generate_candidates import generate_candidates
from select_candidates import select_beneficial_candidates
from edge_difficulty import compute_edge_difficulty, curriculum_training

# Load dataset
file_path = 'bitcoin-otc.csv'  # Replace with your dataset path
G, node2idx, idx2node, pos_adj, neg_adj, edge_index, edge_signs = load_bitcoin_dataset(file_path)

# Train initial SGCN
n_nodes = len(G.nodes())
sgcn = SGCN(n_nodes)
sgcn = train_sgcn(sgcn, edge_index, edge_signs, pos_edge_index, neg_edge_index)

# Generate candidates
thresholds = (0.7, 0.7, 0.3, 0.3)
add_candidates, del_candidates = generate_candidates(sgcn, edge_index, pos_edge_index, neg_edge_index, n_nodes, thresholds)

# Select beneficial candidates
G_aug, filtered_add_candidates = select_beneficial_candidates(G, add_candidates, del_candidates)

# Update edge_index and edge_signs for augmented graph
new_edges = [(node2idx[u], node2idx[v], sign) for u, v, sign in filtered_add_candidates]
new_edge_index = torch.tensor([[u for u, v, s in new_edges], [v for u, v, s in new_edges]], dtype=torch.long)
new_edge_signs = torch.tensor([s for u, v, s in new_edges], dtype=torch.float)
edge_index = torch.cat([edge_index, new_edge_index], dim=1)
edge_signs = torch.cat([edge_signs, new_edge_signs])

# Update positive and negative edge indices
pos_edge_index = edge_index[:, edge_signs == 1]
neg_edge_index = edge_index[:, edge_signs == -1]

# Compute difficulty scores and train with curriculum
difficulty_scores = compute_edge_difficulty(G_aug, edge_index)
sgcn = curriculum_training(sgcn, edge_index, edge_signs, difficulty_scores, pos_edge_index, neg_edge_index)

# Save augmented graph (optional)
nx.write_edgelist(G_aug, 'bitcoin-otc-augmented.edgelist', data=['sign'])