import torch
import networkx as nx
import os
import sys
import numpy as np

# Check for pandas availability and provide helpful error message
try:
    from load_dataset import load_bitcoin_dataset
except ImportError as e:
    if "pandas" in str(e).lower() or "numpy" in str(e).lower():
        print("="*60)
        print("DEPENDENCY ERROR")
        print("="*60)
        print("There's a compatibility issue with pandas/numpy.")
        print("Please run the following command to fix it:")
        print("  python fix_dependencies.py")
        print("\nOr manually install compatible versions:")
        print("  pip uninstall pandas numpy -y")
        print("  pip install \"numpy>=1.24.0,<2.0.0\" \"pandas>=2.0.0,<3.0.0\"")
        print("="*60)
        sys.exit(1)
    else:
        raise e

from sgcn import SGCN, train_sgcn
from generate_candidates import generate_candidates
from select_candidates import select_beneficial_candidates
from edge_difficulty import compute_edge_difficulty, curriculum_training

# Add parent directory to path to import CASDGNN components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import create_adjacency_lists, compute_centrality_features, compute_node_sign_influence, create_adj_matrices
from model import CA_SDGNN
from train_utils import pretrain, finetune
from eval_utils import inference
from fea_extra import FeaExtra
from collections import defaultdict

def convert_augmented_graph_to_edgelist(edge_index, edge_signs, idx2node):
    """
    Convert augmented graph tensors to edge list format for CASDGNN.
    
    Args:
    - edge_index: Tensor of edge indices [2, num_edges]
    - edge_signs: Tensor of edge signs [num_edges]
    - idx2node: Dictionary mapping indices to node IDs
    
    Returns:
    - edge_list: List of tuples (source, target, sign)
    """
    edge_list = []
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i].item()
        tgt_idx = edge_index[1, i].item()
        sign = edge_signs[i].item()
        
        # Convert to actual node IDs
        src_node = idx2node[src_idx]
        tgt_node = idx2node[tgt_idx]
        edge_list.append((src_node, tgt_node, int(sign)))
    
    return edge_list

def analyze_edge_distribution(edge_list, description=""):
    """
    Analyze and print the distribution of positive and negative edges.
    
    Args:
    - edge_list: List of edges (source, target, sign)
    - description: Description for logging
    
    Returns:
    - pos_count, neg_count: Counts of positive and negative edges
    """
    pos_count = sum(1 for _, _, sign in edge_list if sign == 1)
    neg_count = sum(1 for _, _, sign in edge_list if sign == -1)
    total_count = len(edge_list)
    
    print(f"\n{description} Edge Distribution:")
    print(f"  Positive edges: {pos_count:,} ({pos_count/total_count*100:.1f}%)")
    print(f"  Negative edges: {neg_count:,} ({neg_count/total_count*100:.1f}%)")
    print(f"  Total edges: {total_count:,}")
    print(f"  Positive/Negative ratio: {pos_count/neg_count:.3f}" if neg_count > 0 else "  Positive/Negative ratio: inf")
    
    return pos_count, neg_count

def balance_edge_candidates(sgcn, edge_index, pos_edge_index, neg_edge_index, edge_signs, n_nodes, target_ratio=1.0):
    """
    Generate balanced edge candidates to achieve target positive/negative ratio.
    
    Args:
    - sgcn: Trained SGCN model
    - edge_index, pos_edge_index, neg_edge_index: Edge indices
    - edge_signs: Edge signs
    - n_nodes: Number of nodes
    - target_ratio: Target ratio of positive to negative edges (1.0 = equal)
    
    Returns:
    - add_candidates: List of candidate edges to add
    - del_candidates: List of candidate edges to delete
    """
    # First, analyze current distribution
    current_edge_list = [(edge_index[0, i].item(), edge_index[1, i].item(), edge_signs[i].item()) 
                        for i in range(edge_index.shape[1])]
    current_pos, current_neg = analyze_edge_distribution(current_edge_list, "Current Dataset")
    
    # Calculate how many edges we need to add/remove to achieve balance
    total_edges = current_pos + current_neg
    target_pos = int(total_edges * target_ratio / (1 + target_ratio))
    target_neg = total_edges - target_pos
    
    print(f"\nTarget Distribution:")
    print(f"  Target positive edges: {target_pos:,}")
    print(f"  Target negative edges: {target_neg:,}")
    print(f"  Need to add positive edges: {max(0, target_pos - current_pos):,}")
    print(f"  Need to add negative edges: {max(0, target_neg - current_neg):,}")
    
    # Determine what type of edges to add more of
    need_more_pos = target_pos > current_pos
    need_more_neg = target_neg > current_neg
    
    if need_more_pos and need_more_neg:
        # Need both types - use moderate thresholds
        thresholds = (0.7, 0.7, 0.1, 0.1)
        print("Strategy: Adding both positive and negative edges")
    elif need_more_pos:
        # Need more positive edges
        thresholds = (0.6, 1.1, 0.1, 0.1)  # Lower threshold for positive, high for negative
        print("Strategy: Adding positive edges, avoiding negative edges")
    elif need_more_neg:
        # Need more negative edges  
        thresholds = (1.1, 0.6, 0.1, 0.1)  # High threshold for positive, lower for negative
        print("Strategy: Adding negative edges, avoiding positive edges")
    else:
        # Already balanced
        thresholds = (0.8, 0.8, 0.1, 0.1)
        print("Strategy: Maintaining current balance")
    
    print(f"Using thresholds: eps_add_pos={thresholds[0]}, eps_add_neg={thresholds[1]}")
    
    # Import the generate_candidates function
    from generate_candidates import generate_candidates
    add_candidates, del_candidates = generate_candidates(
        sgcn, edge_index, pos_edge_index, neg_edge_index, edge_signs, n_nodes, thresholds
    )
    
    return add_candidates, del_candidates

def filter_candidates_for_balance(add_candidates, current_pos, current_neg, target_ratio=1.0):
    """
    Filter candidates to achieve better balance.
    
    Args:
    - add_candidates: List of candidate edges to add
    - current_pos, current_neg: Current counts of positive and negative edges
    - target_ratio: Target ratio of positive to negative edges
    
    Returns:
    - filtered_candidates: Balanced list of candidates
    """
    total_current = current_pos + current_neg
    target_pos = int(total_current * target_ratio / (1 + target_ratio))
    target_neg = total_current - target_pos
    
    pos_needed = max(0, target_pos - current_pos)
    neg_needed = max(0, target_neg - current_neg)
    
    print(f"\nFiltering candidates for balance:")
    print(f"  Positive edges needed: {pos_needed}")
    print(f"  Negative edges needed: {neg_needed}")
    
    # Separate candidates by sign
    pos_candidates = [cand for cand in add_candidates if len(cand) >= 3 and cand[2] == 1]
    neg_candidates = [cand for cand in add_candidates if len(cand) >= 3 and cand[2] == -1]
    
    print(f"  Available positive candidates: {len(pos_candidates)}")
    print(f"  Available negative candidates: {len(neg_candidates)}")
    
    # Select the needed number of each type
    selected_pos = pos_candidates[:pos_needed] if pos_needed > 0 else []
    selected_neg = neg_candidates[:neg_needed] if neg_needed > 0 else []
    
    filtered_candidates = selected_pos + selected_neg
    
    print(f"  Selected positive candidates: {len(selected_pos)}")
    print(f"  Selected negative candidates: {len(selected_neg)}")
    print(f"  Total filtered candidates: {len(filtered_candidates)}")
    
    return filtered_candidates

def save_edgelist_file(edge_list, filename):
    """
    Save edge list to file in CASDGNN format.
    
    Args:
    - edge_list: List of tuples (source, target, sign)
    - filename: Output filename
    """
    with open(filename, 'w') as f:
        for src, tgt, sign in edge_list:
            f.write(f"{src} {tgt} {sign}\n")

def compute_weight_dict(adj_pos_out, adj_neg_out, fea_extra):
    """
    Compute weight dictionary for CASDGNN fine-tuning.
    """
    weight_dict = defaultdict(dict)
    for node in adj_pos_out:
        for neighbor in adj_pos_out[node]:
            weight_dict[node][neighbor] = 1.0
    for node in adj_neg_out:
        for neighbor in adj_neg_out[node]:
            weight_dict[node][neighbor] = 1.0
    return weight_dict

def get_num_nodes_from_edgelist(edge_list):
    """Get number of nodes from edge list."""
    nodes = set()
    for src, tgt, _ in edge_list:
        nodes.add(src)
        nodes.add(tgt)
    return len(nodes)

def create_node_mapping(edge_list):
    """
    Create mapping from original node IDs to continuous range [0, num_nodes-1].
    
    Returns:
    - node_to_idx: Dict mapping original node ID to new index
    - idx_to_node: Dict mapping new index to original node ID
    - num_nodes: Number of unique nodes
    """
    nodes = set()
    for src, tgt, _ in edge_list:
        nodes.add(src)
        nodes.add(tgt)
    
    nodes = sorted(list(nodes))  # Sort for consistency
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for idx, node in enumerate(nodes)}
    
    return node_to_idx, idx_to_node, len(nodes)

def save_node_mapping(node_to_idx, idx_to_node, filename_prefix='node_mapping'):
    """Save node mapping to files for future reference."""
    import json
    
    # Save node_to_idx mapping
    with open(f'{filename_prefix}_to_idx.json', 'w') as f:
        json.dump(node_to_idx, f, indent=2)
    
    # Save idx_to_node mapping
    with open(f'{filename_prefix}_to_node.json', 'w') as f:
        json.dump(idx_to_node, f, indent=2)
    
    print(f"Node mappings saved to {filename_prefix}_to_idx.json and {filename_prefix}_to_node.json")

def remap_edge_list(edge_list, node_to_idx):
    """
    Remap edge list to use continuous node indices.
    
    Args:
    - edge_list: Original edge list with arbitrary node IDs
    - node_to_idx: Mapping from original node ID to new index
    
    Returns:
    - remapped_edge_list: Edge list with remapped node indices
    """
    remapped_edge_list = []
    for src, tgt, sign in edge_list:
        new_src = node_to_idx[src]
        new_tgt = node_to_idx[tgt]
        remapped_edge_list.append((new_src, new_tgt, sign))
    
    return remapped_edge_list

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*80)
print("CASDGNN Integration with DropEdge Augmentation")
print("="*80)
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Model hyperparameters for CASDGNN
node_feat_dim = 16
embed_dim = 16
centrality_dim = 2
num_heads = 4
num_layers = 2
dropout_rate = 0.1
lr = 0.0005
weight_decay = 0.0001

# Training parameters
pretrain_epochs = 100
finetune_epochs = 50

# Load dataset
print("Loading Bitcoin dataset...")
file_path = r'D:\abhin\Comding\ML\Capstone\casdgnn\dropedge_augmentation\bitcoin_alpha.csv'  # Replace with your dataset path
G, node2idx, idx2node, pos_adj, neg_adj, edge_index, edge_signs, pos_edge_index, neg_edge_index = load_bitcoin_dataset(file_path)

# Analyze original dataset distribution
original_edge_list = [(edge_index[0, i].item(), edge_index[1, i].item(), edge_signs[i].item()) 
                     for i in range(edge_index.shape[1])]
original_pos, original_neg = analyze_edge_distribution(original_edge_list, "Original Dataset")

print("Training initial SGCN model for augmentation...")
# Train initial SGCN for candidate generation
n_nodes = len(G.nodes())
sgcn = SGCN(n_nodes)
sgcn = train_sgcn(sgcn, edge_index, edge_signs, pos_edge_index, neg_edge_index)

print("\nGenerating balanced edge candidates...")
# Generate candidates with balancing strategy
target_ratio = 1.0  # 1:1 ratio of positive to negative edges
add_candidates, del_candidates = balance_edge_candidates(
    sgcn, edge_index, pos_edge_index, neg_edge_index, edge_signs, n_nodes, target_ratio
)

print(f"\nCandidate generation results:")
print(f"  Add candidates: {len(add_candidates)}")
print(f"  Delete candidates: {len(del_candidates)}")

print("\nSelecting beneficial candidates...")
# Select beneficial candidates
G_aug, filtered_add_candidates = select_beneficial_candidates(G, add_candidates, del_candidates)

# Further filter candidates for better balance
filtered_add_candidates = filter_candidates_for_balance(
    filtered_add_candidates, original_pos, original_neg, target_ratio
)

print("\nCreating augmented graph...")
# Update edge_index and edge_signs for augmented graph
new_edges = [(node2idx[u], node2idx[v], sign) for u, v, sign in filtered_add_candidates]
new_edge_index = torch.tensor([[u for u, v, s in new_edges], [v for u, v, s in new_edges]], dtype=torch.long)
new_edge_signs = torch.tensor([s for u, v, s in new_edges], dtype=torch.float)

# Combine original and new edges
edge_index_aug = torch.cat([edge_index, new_edge_index], dim=1)
edge_signs_aug = torch.cat([edge_signs, new_edge_signs])

# Analyze augmented dataset distribution
augmented_edge_list = [(edge_index_aug[0, i].item(), edge_index_aug[1, i].item(), edge_signs_aug[i].item()) 
                      for i in range(edge_index_aug.shape[1])]
aug_pos, aug_neg = analyze_edge_distribution(augmented_edge_list, "Augmented Dataset")

# Print improvement statistics
print(f"\nAugmentation Results:")
print(f"  Added edges: {len(filtered_add_candidates)}")
pos_added = sum(1 for _, _, sign in filtered_add_candidates if sign == 1)
neg_added = sum(1 for _, _, sign in filtered_add_candidates if sign == -1)
print(f"    - Positive edges added: {pos_added}")
print(f"    - Negative edges added: {neg_added}")
print(f"  Original ratio (pos/neg): {original_pos/original_neg:.3f}")
print(f"  Augmented ratio (pos/neg): {aug_pos/aug_neg:.3f}")
print(f"  Balance improvement: {abs(1.0 - aug_pos/aug_neg) < abs(1.0 - original_pos/original_neg)}")

# Save augmented graph
print("Saving augmented graph...")
nx.write_edgelist(G_aug, 'bitcoin-otc-augmented.edgelist', data=['sign'])

# Convert augmented graph to CASDGNN format
print("Converting augmented graph to CASDGNN format...")
augmented_edge_list = convert_augmented_graph_to_edgelist(edge_index_aug, edge_signs_aug, idx2node)

# Create train/test split (80/20 split)
np.random.seed(42)
np.random.shuffle(augmented_edge_list)
split_idx = int(0.8 * len(augmented_edge_list))
train_edge_list = augmented_edge_list[:split_idx]
test_edge_list = augmented_edge_list[split_idx:]

# Analyze train/test distribution
train_pos, train_neg = analyze_edge_distribution(train_edge_list, "Training Split")
test_pos, test_neg = analyze_edge_distribution(test_edge_list, "Test Split")

# Save edge lists for CASDGNN
train_file = 'bitcoin_alpha_augmented_train.edgelist'
test_file = 'bitcoin_alpha_augmented_test.edgelist'
save_edgelist_file(train_edge_list, train_file)
save_edgelist_file(test_edge_list, test_file)

print(f"Saved training edges: {len(train_edge_list)}")
print(f"Saved test edges: {len(test_edge_list)}")

# Now use CASDGNN for final predictions
print("\n" + "="*60)
print("STARTING CASDGNN TRAINING ON AUGMENTED GRAPH")
print("="*60)

# Create node mapping for continuous indices using BOTH train and test data
print("Creating node mapping from all edges...")
all_edge_list = train_edge_list + test_edge_list
node_to_idx, idx_to_node, num_nodes = create_node_mapping(all_edge_list)
print(f"Number of nodes in augmented graph: {num_nodes}")

# Save node mapping for future reference
save_node_mapping(node_to_idx, idx_to_node, 'bitcoin_alpha_augmented_node_mapping')

# Remap edge lists to use continuous indices
print("Remapping edge lists...")
train_edge_list_remapped = remap_edge_list(train_edge_list, node_to_idx)
test_edge_list_remapped = remap_edge_list(test_edge_list, node_to_idx)

print(f"Training edges: {len(train_edge_list_remapped)}")
print(f"Test edges: {len(test_edge_list_remapped)}")

# Verify that all nodes in both datasets are covered
train_nodes = set()
test_nodes = set()
for src, tgt, _ in train_edge_list_remapped:
    train_nodes.add(src)
    train_nodes.add(tgt)
for src, tgt, _ in test_edge_list_remapped:
    test_nodes.add(src)
    test_nodes.add(tgt)

print(f"Training data covers {len(train_nodes)} nodes")
print(f"Test data covers {len(test_nodes)} nodes")
print(f"Union covers {len(train_nodes.union(test_nodes))} nodes")

# Create adjacency lists using remapped edge list
print("Creating adjacency lists...")
adj_pos_out, adj_neg_out, adj_pos_in, adj_neg_in = create_adjacency_lists(train_edge_list_remapped)

# Compute centrality features and node sign influence
# Use full graph for centrality computation but only training data for sign influence
print("Computing centrality features and node sign influence...")
all_edge_list_remapped = train_edge_list_remapped + test_edge_list_remapped
centrality_features = compute_centrality_features(all_edge_list_remapped, num_nodes)  # Use all edges for centrality
node_sign_influence = compute_node_sign_influence(train_edge_list_remapped, num_nodes)  # Use only training for sign influence

# Create adjacency matrices
print("Creating adjacency matrices...")
adj_matrices = create_adj_matrices(adj_pos_out, adj_neg_out, adj_pos_in, adj_neg_in, num_nodes)

# Initialize node features
print("Initializing node features...")
node_features = torch.randn(num_nodes, node_feat_dim, device=device)

# Initialize CASDGNN model
print("Initializing CASDGNN model...")
model = CA_SDGNN(
    node_feat_dim=node_feat_dim,
    embed_dim=embed_dim,
    centrality_dim=centrality_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    device=device,
    dropout_rate=dropout_rate
).to(device)

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Step 1: Pretrain CASDGNN
print(f"\nStep 1: Pretraining CASDGNN for {pretrain_epochs} epochs...")
pretrain_path = 'models/bitcoin_alpha_augmented_pretrained.pth'
pretrain(
    model=model,
    nodes=list(range(num_nodes)),
    node_features=node_features,
    centrality_features=centrality_features,
    adj_matrices=adj_matrices,
    node_sign_influence=node_sign_influence,
    optimizer=optimizer,
    epochs=pretrain_epochs,
    device=device,
    save_path=pretrain_path
)
print(f"Pretrained model saved to {pretrain_path}")

# Step 2: Fine-tune CASDGNN
print(f"\nStep 2: Fine-tuning CASDGNN for {finetune_epochs} epochs...")
# Load pretrained model
model.load_state_dict(torch.load(pretrain_path, map_location=device))

# Prepare data for fine-tuning using remapped edge list
edge_list_for_finetuning = [(u, v) for u, v, _ in train_edge_list_remapped]
labels = torch.tensor([1 if sign == 1 else 0 for _, _, sign in train_edge_list_remapped], dtype=torch.float32).to(device)

# Compute weight dictionary
fea_extra = FeaExtra('bitcoin_alpha', k=1)
weight_dict = compute_weight_dict(adj_pos_out, adj_neg_out, fea_extra)

finetune_path = 'models/bitcoin_alpha_augmented_finetuned.pth'
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
    edge_list=edge_list_for_finetuning,
    labels=labels,
    weight_dict=weight_dict,
    optimizer=optimizer,
    epochs=finetune_epochs,
    device=device,
    save_path=finetune_path
)
print(f"Fine-tuned model saved to {finetune_path}")

# Step 3: Inference on test data
print(f"\nStep 3: Running inference on test data...")
test_edge_list_for_pred = [(u, v) for u, v, _ in test_edge_list_remapped]
test_labels = torch.tensor([1 if sign == 1 else 0 for _, _, sign in test_edge_list_remapped], dtype=torch.float32).to(device)

output_dir = 'results/output_augmented'
preds, true_labels, metrics = inference(
    model=model,
    nodes=list(range(num_nodes)),
    node_features=node_features,
    centrality_features=centrality_features,
    adj_matrices=adj_matrices,
    node_sign_influence=node_sign_influence,
    edge_list=test_edge_list_for_pred,
    test_labels=test_labels,
    model_path=finetune_path,
    output_dir=output_dir,
    device=device
)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Original nodes in full dataset: {len(set(src for src, _, _ in train_edge_list + test_edge_list).union(set(tgt for _, tgt, _ in train_edge_list + test_edge_list)))}")
print(f"Nodes used for training: {len(train_nodes)}")
print(f"Nodes used for testing: {len(test_nodes)}")
print(f"Total mapped nodes: {num_nodes}")
print(f"Training edges: {len(train_edge_list_remapped)}")
print(f"Test edges: {len(test_edge_list_remapped)}")
print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

print(f"\nPredictions saved to {output_dir}/predictions.txt")
print("Note: Predictions use remapped node indices [0, num_nodes-1]")
print("Use the idx_to_node mapping to convert back to original node IDs")
print("Augmented graph training and evaluation completed!")