import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

def pretrain(model, nodes, node_features, centrality_features, adj_matrices, node_sign_influence, optimizer, epochs, device, save_path):
    """
    Unsupervised pretraining for graph reconstruction.
    
    Args:
    - model: CA_SDGNN model instance.
    - nodes: List or tensor of node indices.
    - node_features: Node feature embedding module.
    - centrality_features: Dict of centrality feature tensors.
    - adj_matrices: List of adjacency matrices for different relations.
    - node_sign_influence: Tensor of node sign influence values.
    - optimizer: Optimizer for training.
    - epochs: Number of pretraining epochs.
    - device: Device to run the model on.
    - save_path: Path to save the pretrained model.
    """
    model.train()
    nodes = torch.tensor(nodes, dtype=torch.long).to(device)
    adj_matrices = [adj.to(device) for adj in adj_matrices]
    node_sign_influence = node_sign_influence.to(device)
    centrality_features = {k: v.to(device) for k, v in centrality_features.items()}
    
    for epoch in tqdm(range(epochs), desc="Pretraining Epochs"):
        optimizer.zero_grad()
        output_embeddings = model(nodes, node_features, centrality_features, adj_matrices, node_sign_influence)
        reconstructed_adj = torch.mm(output_embeddings, output_embeddings.t())
        target_adj = sum(adj.to_dense() for adj in adj_matrices) / len(adj_matrices)
        loss = F.mse_loss(reconstructed_adj, target_adj)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Pretraining Loss: {loss.item():.4f}")
    
    # Create directory if save_path includes a directory
    save_dir = os.path.dirname(save_path)
    if save_dir:  # Only create directory if save_dir is not empty
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Pretrained model saved to {save_path}")

def finetune(model, nodes, node_features, centrality_features, adj_matrices, node_sign_influence, pos_out_neighbors, neg_out_neighbors, pos_in_neighbors, neg_in_neighbors, edge_list, labels, weight_dict, optimizer, epochs, device, save_path):
    """
    Supervised fine-tuning for link sign prediction.
    
    Args:
    - model: CA_SDGNN model instance.
    - nodes: List or tensor of node indices.
    - node_features: Node feature embedding module.
    - centrality_features: Dict of centrality feature tensors.
    - adj_matrices: List of adjacency matrices for different relations.
    - node_sign_influence: Tensor of node sign influence values.
    - pos_out_neighbors, neg_out_neighbors, pos_in_neighbors, neg_in_neighbors: Adjacency lists.
    - edge_list: List of edges for link prediction.
    - labels: Tensor of edge labels (0 or 1).
    - weight_dict: Dictionary of edge weights for loss computation.
    - optimizer: Optimizer for training.
    - epochs: Number of fine-tuning epochs.
    - device: Device to run the model on.
    - save_path: Path to save the fine-tuned model.
    """
    model.train()
    nodes = torch.tensor(nodes, dtype=torch.long).to(device)
    adj_matrices = [adj.to(device) for adj in adj_matrices]
    node_sign_influence = node_sign_influence.to(device)
    centrality_features = {k: v.to(device) for k, v in centrality_features.items()}
    labels = labels.to(device)
    
    for epoch in tqdm(range(epochs), desc="Fine-tuning Epochs"):
        optimizer.zero_grad()
        
        # First call forward to initialize the model's internal attributes
        model(nodes, node_features, centrality_features, adj_matrices, node_sign_influence)
        
        loss = model.criterion(
            nodes.tolist(),
            pos_out_neighbors,
            neg_out_neighbors,
            pos_in_neighbors,
            neg_in_neighbors,
            edge_list,
            labels,
            weight_dict
        )
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Fine-tuning Loss: {loss.item():.4f}")
    
    # Create directory if save_path includes a directory
    save_dir = os.path.dirname(save_path)
    if save_dir:  # Only create directory if save_dir is not empty
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved to {save_path}")