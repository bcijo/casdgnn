import torch
import networkx as nx
from collections import defaultdict

def load_edge_list(filename):
    """
    Load the edge list from a file.
    
    Args:
    - filename: Path to the edge list file.
    
    Returns:
    - edge_list: List of tuples (source, target, sign).
    """
    edge_list = []
    with open(filename, 'r') as f:
        for line in f:
            u, v, sign = map(int, line.strip().split())
            edge_list.append((u, v, sign))
    return edge_list

def get_num_nodes(edge_list):
    """
    Determine the number of nodes from the edge list.
    
    Args:
    - edge_list: List of edges (source, target, sign).
    
    Returns:
    - num_nodes: The number of nodes in the graph.
    """
    nodes = set()
    for u, v, _ in edge_list:
        nodes.add(u)
        nodes.add(v)
    return max(nodes) + 1 if nodes else 0

def create_adjacency_lists(edge_list):
    """
    Create adjacency lists for different relation types.
    
    Args:
    - edge_list: List of edges (source, target, sign).
    
    Returns:
    - adj_pos_out: Dict where key is node, value is list of positive outgoing neighbors.
    - adj_neg_out: Dict where key is node, value is list of negative outgoing neighbors.
    - adj_pos_in: Dict where key is node, value is list of positive incoming neighbors.
    - adj_neg_in: Dict where key is node, value is list of negative incoming neighbors.
    """
    adj_pos_out = defaultdict(list)
    adj_neg_out = defaultdict(list)
    adj_pos_in = defaultdict(list)
    adj_neg_in = defaultdict(list)
    for u, v, sign in edge_list:
        if sign == 1:
            adj_pos_out[u].append(v)
            adj_pos_in[v].append(u)
        else:
            adj_neg_out[u].append(v)
            adj_neg_in[v].append(u)
    return adj_pos_out, adj_neg_out, adj_pos_in, adj_neg_in

def compute_centrality_features(edge_list, num_nodes):
    """
    Compute centrality features (betweenness and closeness) for the graph.
    
    Args:
    - edge_list: List of edges (source, target, sign).
    - num_nodes: Number of nodes in the graph.
    
    Returns:
    - centrality_features: Dict with keys "betweenness" and "closeness", each mapping to a tensor of shape (num_nodes,).
    """
    # Create an undirected graph for centrality computation
    graph = nx.Graph()
    for u, v, _ in edge_list:
        graph.add_edge(u, v)
    
    # Compute betweenness and closeness centrality
    betweenness = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    
    # Create tensors for centrality features
    centrality_features = {
        "betweenness": torch.tensor([betweenness.get(i, 0) for i in range(num_nodes)], dtype=torch.float32),
        "closeness": torch.tensor([closeness.get(i, 0) for i in range(num_nodes)], dtype=torch.float32)
    }
    return centrality_features

def compute_node_sign_influence(edge_list, num_nodes):
    """
    Compute node sign influence based on the edge list.
    
    Args:
    - edge_list: List of edges (source, target, sign).
    - num_nodes: Number of nodes in the graph.
    
    Returns:
    - node_sign_influence: Tensor of shape (num_nodes,) with values between -1 and 1.
    """
    pos_count = torch.zeros(num_nodes, dtype=torch.float32)
    neg_count = torch.zeros(num_nodes, dtype=torch.float32)
    for u, v, sign in edge_list:
        if sign == 1:
            pos_count[u] += 1
            pos_count[v] += 1
        else:
            neg_count[u] += 1
            neg_count[v] += 1
    total_count = pos_count + neg_count
    net_influence = (pos_count - neg_count) / (total_count + 1e-6)
    net_influence = torch.tanh(net_influence)
    degree_scaled = total_count / (total_count.max() + 1e-6)
    node_sign_influence = net_influence + torch.sign(net_influence) * degree_scaled
    node_sign_influence = torch.clamp(node_sign_influence, -1, 1)
    return node_sign_influence

def create_signed_adj_matrix(edge_list, num_nodes):
    """
    Create a signed adjacency matrix from the edge list.
    
    Args:
    - edge_list: List of edges (source, target, sign).
    - num_nodes: Number of nodes in the graph.
    
    Returns:
    - signed_adj_matrix: A sparse tensor representing the signed adjacency matrix.
    """
    sources, targets, signs = [], [], []
    for u, v, sign in edge_list:
        sources.append(u)
        targets.append(v)
        signs.append(sign)
    
    indices = torch.LongTensor([sources, targets])
    values = torch.FloatTensor(signs)
    signed_adj_matrix = torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes))
    return signed_adj_matrix

def create_adj_matrices(adj_pos_out, adj_neg_out, adj_pos_in, adj_neg_in, num_nodes):
    """
    Create adjacency matrices for different relation types.
    
    Args:
    - adj_pos_out: Dict where key is node, value is list of positive outgoing neighbors.
    - adj_neg_out: Dict where key is node, value is list of negative outgoing neighbors.
    - adj_pos_in: Dict where key is node, value is list of positive incoming neighbors.
    - adj_neg_in: Dict where key is node, value is list of negative incoming neighbors.
    - num_nodes: Number of nodes in the graph.
    
    Returns:
    - List of sparse tensors representing adjacency matrices for different relations.
    """
    relations = [adj_pos_out, adj_neg_out, adj_pos_in, adj_neg_in]
    adj_matrices = []
    
    for relation in relations:
        sources, targets = [], []
        for src in relation:
            for tgt in relation[src]:
                sources.append(src)
                targets.append(tgt)
        
        if sources and targets:
            indices = torch.LongTensor([sources, targets])
            values = torch.ones(len(sources))
            adj_matrix = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
            adj_matrices.append(adj_matrix)
        else:
            # Create an empty sparse tensor if there are no edges for this relation
            adj_matrix = torch.sparse_coo_tensor(
                torch.LongTensor([[],[]]), 
                torch.FloatTensor([]),
                (num_nodes, num_nodes)
            )
            adj_matrices.append(adj_matrix)
    
    return adj_matrices