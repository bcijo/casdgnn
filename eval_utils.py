import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

def compute_metrics(preds, true_labels):
    """
    Compute evaluation metrics with optimal threshold for binary predictions.
    
    Args:
    - preds: Tensor of predicted logits.
    - true_labels: Tensor of true labels (0 or 1).
    
    Returns:
    - metrics: Dict containing accuracy, precision, recall, binary F1, micro F1, macro F1, AUC, and optimal threshold.
    """
    preds_cpu = preds.detach().cpu()
    true_labels_cpu = true_labels.detach().cpu()
    probs = torch.sigmoid(preds_cpu).numpy()
    true_labels_np = true_labels_cpu.numpy()
    
    # Find optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []
    for threshold in thresholds:
        binary_preds = (probs > threshold).astype(int)
        f1 = f1_score(true_labels_np, binary_preds, average='binary')
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute metrics with optimal threshold
    binary_preds = (probs > optimal_threshold).astype(int)
    acc = accuracy_score(true_labels_np, binary_preds)
    prec = precision_score(true_labels_np, binary_preds)
    rec = recall_score(true_labels_np, binary_preds)
    binary_f1 = f1_score(true_labels_np, binary_preds, average='binary')
    micro_f1 = f1_score(true_labels_np, binary_preds, average='micro')
    macro_f1 = f1_score(true_labels_np, binary_preds, average='macro')
    auc = roc_auc_score(true_labels_np, probs)
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "binary_f1": binary_f1,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "auc": auc,
        "optimal_threshold": optimal_threshold
    }
    
    print(f"Optimal threshold: {optimal_threshold:.2f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Binary F1 Score: {binary_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    return metrics

def inference(model, nodes, node_features, centrality_features, adj_matrices, node_sign_influence, edge_list, test_labels, model_path, output_dir, device):
    """
    Perform inference and save predictions.
    
    Args:
    - model: CA_SDGNN model instance.
    - nodes: List or tensor of node indices.
    - node_features: Node feature embedding module.
    - centrality_features: Dict of centrality feature tensors.
    - adj_matrices: List of adjacency matrices for different relations.
    - node_sign_influence: Tensor of node sign influence values.
    - edge_list: List of edges for prediction.
    - test_labels: Tensor of true labels for evaluation.
    - model_path: Path to the trained model weights.
    - output_dir: Directory to save predictions.
    - device: Device to run the model on.
    
    Returns:
    - preds: Predicted logits.
    - true_labels: True labels.
    - metrics: Computed evaluation metrics.
    """
    # Load model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Move inputs to device
    nodes = torch.tensor(nodes, dtype=torch.long).to(device)
    node_features = node_features.to(device)
    centrality_features = {k: v.to(device) for k, v in centrality_features.items()}
    adj_matrices = [adj.to(device) for adj in adj_matrices]
    node_sign_influence = node_sign_influence.to(device)
    test_labels = test_labels.to(device)
    
    # Inference
    with torch.no_grad():
        preds = model(nodes, node_features, centrality_features, adj_matrices, node_sign_influence, edge_list)
    
    # Compute metrics
    metrics = compute_metrics(preds, test_labels)
    
    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'predictions.txt')
    preds_np = torch.sigmoid(preds).cpu().numpy()
    labels_np = test_labels.cpu().numpy()
    with open(output_file, 'w') as f:
        f.write("Source\tTarget\tPrediction\tTrueLabel\n")
        for (src, tgt), pred, label in zip(edge_list, preds_np, labels_np):
            f.write(f"{src}\t{tgt}\t{pred:.4f}\t{int(label)}\n")
    print(f"Predictions saved to {output_file}")
    
    return preds, test_labels, metrics