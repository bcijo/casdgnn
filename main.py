import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from data_utils import load_edge_list, get_num_nodes, create_adjacency_lists, compute_centrality_features, compute_node_sign_influence, create_signed_adj_matrix, create_adj_matrices
from model import CA_SDGNN
from train_utils import pretrain, finetune
from eval_utils import inference
from fea_extra import FeaExtra
from collections import defaultdict
import os

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Hybrid CA-SDGNN Model")
    parser.add_argument('--dataset', type=str, default='bitcoin_alpha', help='Dataset name')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune', 'infer'], required=True, help='Mode: pretrain, finetune, or infer')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training edge list file')
    parser.add_argument('--test_file', type=str, help='Path to test edge list file (required for infer mode)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--node_feat_dim', type=int, default=16, help='Node feature dimension')
    parser.add_argument('--embed_dim', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--centrality_dim', type=int, default=2, help='Centrality feature dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size')
    parser.add_argument('--pretrain_path', type=str, default=None, help='Path to save/load pretrained model')
    parser.add_argument('--finetune_path', type=str, default=None, help='Path to save/load fine-tuned model')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save predictions')
    return parser.parse_args()

def set_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_weight_dict(adj_pos_out, adj_neg_out, fea_extra):
    """
    Compute weight_dict for fine-tuning loss based on triangle features.
    
    Args:
        adj_pos_out (dict): Positive outgoing adjacency list.
        adj_neg_out (dict): Negative outgoing adjacency list.
        fea_extra (FeaExtra): Instance of FeaExtra for feature extraction.
    
    Returns:
        defaultdict: Nested dictionary with weights for each edge.
    """
    weight_dict = defaultdict(dict)
    mask_pos = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]
    mask_neg = [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
    
    for u in adj_pos_out:
        for v in adj_pos_out[u]:
            v_list = fea_extra.feature_part2(u, v)
            weight = np.dot(v_list, mask_pos)
            weight_dict[u][v] = weight
    
    for u in adj_neg_out:
        for v in adj_neg_out[u]:
            v_list = fea_extra.feature_part2(u, v)
            weight = np.dot(v_list, mask_neg)
            weight_dict[u][v] = weight
    
    return weight_dict

def main():
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() and 'cuda' in torch.device('cuda').type else 'cpu'} (torch.cuda.is_available()={torch.cuda.is_available()})")
    print("[LOG] Parsing arguments...")
    args = parse_args()
    print("[LOG] Setting random seeds...")
    set_seeds(args.seed)
    if args.pretrain_path is None:
        args.pretrain_path = f'{args.dataset}_pretrained_model.pth'
    if args.finetune_path is None:
        args.finetune_path = f'{args.dataset}_finetuned_model.pth'
    print("[LOG] Loading training edge list...")
    train_edge_list = load_edge_list(args.train_file)
    num_nodes = get_num_nodes(train_edge_list)
    print(f"[LOG] Number of nodes: {num_nodes}")
    print("[LOG] Creating adjacency lists...")
    adj_pos_out, adj_neg_out, adj_pos_in, adj_neg_in = create_adjacency_lists(train_edge_list)
    print("[LOG] Computing centrality features and node sign influence...")
    centrality_features = compute_centrality_features(train_edge_list, num_nodes)
    node_sign_influence = compute_node_sign_influence(train_edge_list, num_nodes)
    print("[LOG] Creating adjacency matrices...")
    adj_matrices = create_adj_matrices(adj_pos_out, adj_neg_out, adj_pos_in, adj_neg_in, num_nodes)
    signed_adj_matrix = create_signed_adj_matrix(train_edge_list, num_nodes)
    print("[LOG] Initializing node features...")
    node_features = torch.randn(num_nodes, args.node_feat_dim, device=args.device)
    print("[LOG] Initializing model...")
    model = CA_SDGNN(
        node_feat_dim=args.node_feat_dim,
        embed_dim=args.embed_dim,
        centrality_dim=args.centrality_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        device=args.device,
        dropout_rate=args.dropout_rate
    ).to(args.device)
    print("[LOG] Setting up optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.mode == 'pretrain':
        print("[LOG] Starting pretraining phase...")
        pretrain(
            model=model,
            nodes=list(range(num_nodes)),
            node_features=node_features,
            centrality_features=centrality_features,
            adj_matrices=adj_matrices,
            node_sign_influence=node_sign_influence,
            optimizer=optimizer,
            epochs=args.epochs,
            device=args.device,
            save_path=args.pretrain_path
        )
        print(f"[LOG] Pretrained model saved to {args.pretrain_path}")
    elif args.mode == 'finetune':
        print("[LOG] Starting fine-tuning phase...")
        if os.path.exists(args.pretrain_path):
            print(f"[LOG] Loading pretrained model from {args.pretrain_path}...")
            model.load_state_dict(torch.load(args.pretrain_path, map_location=args.device))
            print(f"[LOG] Loaded pretrained model from {args.pretrain_path}")
        else:
            print(f"[LOG] No pretrained model found at {args.pretrain_path}. Starting fine-tuning from scratch.")
        print("[LOG] Preparing edge list and labels for fine-tuning...")
        edge_list = [(u, v) for u, v, _ in train_edge_list]
        labels = torch.tensor([1 if sign == 1 else 0 for _, _, sign in train_edge_list], dtype=torch.float32).to(args.device)
        print("[LOG] Computing weight_dict for fine-tuning loss...")
        fea_extra = FeaExtra(args.dataset, k=1)
        weight_dict = compute_weight_dict(adj_pos_out, adj_neg_out, fea_extra)
        print("[LOG] Calling finetune()...")
        finetune(
            model=model,
            nodes=list(range(num_nodes)),
            node_features=node_features,
            centrality_features=centrality_features,
            adj_matrices=adj_matrices,
            node_sign_influence=node_sign_influence,
            pos_out_neighbors=adj_pos_out,
            neg_out_neighbors=adj_neg_out,
            pos_in_neighbors=adj_pos_in,
            neg_in_neighbors=adj_neg_in,
            edge_list=edge_list,
            labels=labels,
            weight_dict=weight_dict,
            optimizer=optimizer,
            epochs=args.epochs,
            device=args.device,
            save_path=args.finetune_path
        )
        print(f"[LOG] Fine-tuned model saved to {args.finetune_path}")
    elif args.mode == 'infer':
        print("[LOG] Starting inference phase...")
        if not args.test_file:
            raise ValueError("Test file must be provided for inference mode.")        
        if os.path.exists(args.finetune_path):
            print(f"[LOG] Loading fine-tuned model from {args.finetune_path}...")
            model.load_state_dict(torch.load(args.finetune_path, map_location=args.device))
            print(f"[LOG] Loaded fine-tuned model from {args.finetune_path}")
        else:
            raise FileNotFoundError(f"No fine-tuned model found at {args.finetune_path}")
        print("[LOG] Loading and preparing test edge list...")
        test_edge_list = load_edge_list(args.test_file)
        test_edge_list_for_pred = [(u, v) for u, v, _ in test_edge_list]
        test_labels = torch.tensor([1 if sign == 1 else 0 for _, _, sign in test_edge_list], dtype=torch.float32).to(args.device)
        print("[LOG] Running inference...")
        preds, true_labels, metrics = inference(
            model=model,
            nodes=list(range(num_nodes)),
            node_features=node_features,
            centrality_features=centrality_features,
            adj_matrices=adj_matrices,
            node_sign_influence=node_sign_influence,
            edge_list=test_edge_list_for_pred,
            test_labels=test_labels,
            model_path=args.finetune_path,
            output_dir=args.output_dir,
            device=args.device
        )
        print("Evaluation Metrics:", metrics)

if __name__ == "__main__":
    main()