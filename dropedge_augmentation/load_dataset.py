import pandas as pd
import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix

def load_bitcoin_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    # Load edgelist
    df = pd.read_csv(file_path, names=['source', 'target', 'sign'])
    print(f"Loaded {len(df)} edges from dataset")
    
    # Create NetworkX directed graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(str(row['source']), str(row['target']), sign=row['sign'])
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Get node mapping (NetworkX uses arbitrary node IDs, map to 0-based indices)
    nodes = sorted(G.nodes())
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    idx2node = {idx: node for node, idx in node2idx.items()}
    
    # Create adjacency matrices for positive and negative edges
    n_nodes = len(nodes)
    pos_rows, pos_cols = [], []
    neg_rows, neg_cols = [], []
    
    for u, v, data in G.edges(data=True):
        i, j = node2idx[u], node2idx[v]
        if data['sign'] == 1 or data['sign'] == '1':  # Handle both int and string cases
            pos_rows.append(i)
            pos_cols.append(j)
        elif data['sign'] == -1 or data['sign'] == '-1':  # Handle both int and string cases
            neg_rows.append(i)
            neg_cols.append(j)
        else:
            print(f"Warning: Unexpected sign value: {data['sign']} for edge {u}->{v}")
    
    print(f"Found {len(pos_rows)} positive edges (+1) and {len(neg_rows)} negative edges (-1)")
    
    # Positive adjacency matrix
    pos_adj = csr_matrix((np.ones(len(pos_rows)), (pos_rows, pos_cols)), shape=(n_nodes, n_nodes))
    # Negative adjacency matrix
    neg_adj = csr_matrix((np.ones(len(neg_rows)), (neg_rows, neg_cols)), shape=(n_nodes, n_nodes))
    print("Created sparse adjacency matrices")
    
    # Convert to PyTorch tensors
    pos_edge_index = torch.tensor(np.array(pos_adj.nonzero()), dtype=torch.long)
    neg_edge_index = torch.tensor(np.array(neg_adj.nonzero()), dtype=torch.long)
    print(f"Converted to PyTorch tensors: pos_edges={pos_edge_index.size(1)}, neg_edges={neg_edge_index.size(1)}")
    
    # Edge signs for training
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_signs = torch.cat([torch.ones(pos_edge_index.size(1)), -torch.ones(neg_edge_index.size(1))])
    
    return G, node2idx, idx2node, pos_adj, neg_adj, edge_index, edge_signs, pos_edge_index, neg_edge_index

# Example usage
file_path = 'bitcoin_alpha.csv'  # Replace with your dataset path
G, node2idx, idx2node, pos_adj, neg_adj, edge_index, edge_signs, pos_edge_index, neg_edge_index = load_bitcoin_dataset(file_path)