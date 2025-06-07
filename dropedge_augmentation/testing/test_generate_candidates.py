# filepath: d:\abhin\Comding\ML\Capstone\casdgnn\dropedge_augmentation\testing\test_generate_candidates.py
import torch
import logging
import sys
import os

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_candidates import generate_candidates

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock model class for testing
class MockSGCN:
    def __init__(self):
        pass
    
    def eval(self):
        pass
    
    def __call__(self, pos_edge_index, neg_edge_index):
        # Return random embeddings for testing
        n_nodes = max(pos_edge_index.max(), neg_edge_index.max()) + 1
        return torch.randn(n_nodes, 64)  # 64-dimensional embeddings
    
    def predict_edge(self, z, edge_pairs):
        # Return random logits for 3 classes (pos, neg, no_edge)
        n_pairs = edge_pairs.shape[1]
        return torch.randn(n_pairs, 3)

def test_generate_candidates():
    # Create test data
    logger.info("Setting up test data...")
    n_nodes = 10

    # Create some test edges
    pos_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    neg_edge_index = torch.tensor([[0, 1], [4, 5]], dtype=torch.long)

    # Combine all edges
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_signs = torch.tensor([1, 1, 1, 1, -1, -1], dtype=torch.long)  # Signs for all edges

    logger.info(f"Created test graph with {n_nodes} nodes and {edge_index.shape[1]} edges")
    logger.info(f"Positive edges: {pos_edge_index.shape[1]}, Negative edges: {neg_edge_index.shape[1]}")

    # Create mock model
    sgcn = MockSGCN()

    # Set thresholds
    thresholds = (0.7, 0.7, 0.3, 0.3)  # (eps_add_pos, eps_add_neg, eps_del_pos, eps_del_neg)

    # Generate candidates
    try:
        add_candidates, del_candidates = generate_candidates(
            sgcn, edge_index, pos_edge_index, neg_edge_index, 
            edge_signs, n_nodes, thresholds
        )
        
        logger.info("=== RESULTS ===")
        logger.info(f"Addition candidates: {len(add_candidates)}")
        for i, (u, v, sign) in enumerate(add_candidates[:5]):  # Show first 5
            logger.info(f"  Add edge ({u}, {v}) with sign {sign}")
        if len(add_candidates) > 5:
            logger.info(f"  ... and {len(add_candidates) - 5} more")
        
        logger.info(f"Deletion candidates: {len(del_candidates)}")
        for i, (u, v, sign) in enumerate(del_candidates[:5]):  # Show first 5
            logger.info(f"  Delete edge ({u}, {v}) with sign {sign}")
        if len(del_candidates) > 5:
            logger.info(f"  ... and {len(del_candidates) - 5} more")
            
        return True
    except Exception as e:
        logger.error(f"Error during candidate generation: {e}")
        raise

if __name__ == "__main__":
    test_generate_candidates()