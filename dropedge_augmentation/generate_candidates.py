import torch
import torch.nn.functional as F
import numpy as np
from itertools import combinations
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_candidates(model, edge_index, pos_edge_index, neg_edge_index, edge_signs, n_nodes, thresholds):
    """
    Generate candidate edges for addition and deletion based on model predictions.
    
    Args:
        model: Trained SGCN model
        edge_index: Current edge indices [2, num_edges]
        pos_edge_index: Positive edge indices for training
        neg_edge_index: Negative edge indices for training
        edge_signs: Signs of existing edges (+1 or -1)
        n_nodes: Number of nodes in the graph
        thresholds: Tuple of (eps_add_pos, eps_add_neg, eps_del_pos, eps_del_neg)
    
    Returns:
        add_candidates: List of edges to add [(i, j, sign), ...]
        del_candidates: List of edges to delete [(i, j, sign), ...]
    """
    logger.info("Starting candidate generation...")
    model.eval()
    
    with torch.no_grad():
        # Get node embeddings
        z = model(pos_edge_index, neg_edge_index)
        logger.info(f"Generated embeddings with shape: {z.shape}")
        
        add_candidates = []
        del_candidates = []
        
        # Generate all possible node pairs
        all_pairs = list(combinations(range(n_nodes), 2))
        existing_edges = set(tuple(edge) for edge in edge_index.t().tolist())
        logger.info(f"Total possible pairs: {len(all_pairs)}, Existing edges: {len(existing_edges)}")
        
        # Predict probabilities for all pairs
        pair_tensor = torch.tensor(all_pairs, dtype=torch.long).t()
        logits = model.predict_edge(z, pair_tensor)
        probs = F.softmax(logits, dim=1)  # [N_pairs, 3] (pos, neg, no edge)
        logger.info(f"Computed probabilities for {len(all_pairs)} pairs")
        
        # Thresholds
        eps_add_pos, eps_add_neg, eps_del_pos, eps_del_neg = thresholds
        logger.info(f"Using thresholds: add_pos={eps_add_pos}, add_neg={eps_add_neg}, del_pos={eps_del_pos}, del_neg={eps_del_neg}")
        
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
        
        logger.info(f"Generated {len(add_candidates)} addition candidates and {len(del_candidates)} deletion candidates")
        return add_candidates, del_candidates