import torch
import numpy as np
from itertools import combinations

def generate_candidates(model, edge_index, pos_edge_index, neg_edge_index, n_nodes, thresholds):
    model.eval()
    with torch.no_grad():
        z = model(pos_edge_index, neg_edge_index)
        add_candidates = []
        del_candidates = []
        
        # Generate all possible node pairs
        all_pairs = list(combinations(range(n_nodes), 2))
        existing_edges = set(tuple(edge) for edge in edge_index.t().tolist())
        
        # Predict probabilities for all pairs
        pair_tensor = torch.tensor(all_pairs, dtype=torch.long).t()
        logits = model.predict_edge(z, pair_tensor)
        probs = F.softmax(logits, dim=1)  # [N_pairs, 3] (pos, neg, no edge)
        
        # Thresholds
        eps_add_pos, eps_add_neg, eps_del_pos, eps_del_neg = thresholds
        
        for idx, (i, j) in enumerate(all_pairs):
            pos_prob, neg_prob, _ = probs[idx]
            edge = (i, j)
            
            # Adding edges
            if edge not in existing_edges:
                if pos_prob > eps_add_pos:
                    add_candidates.append((i, j, 1))
                elif neg_prob > eps_add_neg:
                    add_candidates.append((i, j, -1))
            
            # Deleting edges
            if edge in existing_edges:
                edge_idx = edge_index[:, (edge_index[0] == i) & (edge_index[1] == j)]
                if edge_idx.numel() > 0:
                    sign = edge_signs[(edge_index[0] == i) & (edge_index[1] == j)].item()
                    if sign == 1 and pos_prob < eps_del_pos:
                        del_candidates.append((i, j, 1))
                    elif sign == -1 and neg_prob < eps_del_neg:
                        del_candidates.append((i, j, -1))
        
        return add_candidates, del_candidates

# Example usage
thresholds = (0.7, 0.7, 0.3, 0.3)  # Example thresholds (eps_add_pos, eps_add_neg, eps_del_pos, eps_del_neg)
add_candidates, del_candidates = generate_candidates(sgcn, edge_index, pos_edge_index, neg_edge_index, n_nodes, thresholds)