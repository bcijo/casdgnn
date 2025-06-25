# DropEdge Augmentation with CASDGNN Integration

This module implements a hybrid approach that combines the graph augmentation capabilities of SGCN with the advanced prediction capabilities of CASDGNN.

## Pipeline Overview

The integrated pipeline consists of the following steps:

### Phase 1: Graph Augmentation (Using SGCN)
1. **Initial Training**: Train an SGCN model on the original signed graph
2. **Candidate Generation**: Use the trained SGCN to predict potential edge additions and deletions
3. **Candidate Selection**: Filter candidates to preserve structural balance and avoid unbalanced triangles
4. **Graph Augmentation**: Create an augmented graph with beneficial edge modifications

### Phase 2: Final Prediction (Using CASDGNN)
5. **Data Preparation**: Convert augmented graph to CASDGNN-compatible format
6. **Pretraining**: Train CASDGNN on the augmented graph using self-supervised learning
7. **Fine-tuning**: Fine-tune CASDGNN for link sign prediction task
8. **Inference**: Generate final predictions using the trained CASDGNN model

## Key Advantages

- **Best of Both Worlds**: Leverages SGCN's graph augmentation capabilities and CASDGNN's superior prediction performance
- **Structural Balance**: Maintains graph balance properties through careful candidate selection
- **Advanced Features**: CASDGNN incorporates centrality features and attention mechanisms for better predictions
- **Curriculum Learning**: Optional curriculum learning strategy for improved training

## File Descriptions

*   **`main.py`**: The integrated main script that orchestrates the entire workflow including CASDGNN training
*   **`run_casdgnn_integration.py`**: Simple runner script for the integration
*   **`load_dataset.py`**: Contains functions to load signed graph datasets from CSV files
*   **`sgcn.py`**: Defines the SGCN model for graph augmentation phase
*   **`generate_candidates.py`**: Generates candidate edges for addition/deletion using SGCN predictions
*   **`select_candidates.py`**: Filters candidates to maintain structural balance
*   **`edge_difficulty.py`**: Computes edge difficulty scores and implements curriculum learning

## Model Configuration

The CASDGNN model in the integration uses the following default hyperparameters:

- **node_feat_dim**: 16 (Initial node feature dimension)
- **embed_dim**: 16 (Node embedding dimension)
- **centrality_dim**: 2 (Centrality feature dimension)
- **num_heads**: 4 (Number of attention heads)
- **num_layers**: 2 (Number of transformer layers)
- **dropout_rate**: 0.1 (Dropout rate)
- **lr**: 0.0005 (Learning rate)
- **weight_decay**: 0.0001 (L2 regularization)

## Training Configuration

- **Pretraining epochs**: 100
- **Fine-tuning epochs**: 50
- **Train/Test split**: 80/20

## Running the Integration

### Prerequisites
Ensure you have all required dependencies installed and the Bitcoin Alpha dataset available.

**Important**: If you encounter dependency issues (especially NumPy/pandas compatibility), see `SETUP.md` for detailed setup instructions.

### Quick Setup and Test
```bash
cd dropedge_augmentation
python fix_dependencies.py  # Fix any compatibility issues
python test_core.py         # Test core components
python test_integration.py  # Test full integration
```

### Method 1: Direct Execution
```bash
cd dropedge_augmentation
python main.py
```

### Method 2: Using Runner Script
```bash
cd dropedge_augmentation
python run_casdgnn_integration.py
```

## Output Files

The integration will generate several output files:

1. **bitcoin-otc-augmented.edgelist**: The augmented graph in NetworkX format
2. **bitcoin_alpha_augmented_train.edgelist**: Training edges for CASDGNN
3. **bitcoin_alpha_augmented_test.edgelist**: Test edges for CASDGNN
4. **models/bitcoin_alpha_augmented_pretrained.pth**: Pretrained CASDGNN model
5. **models/bitcoin_alpha_augmented_finetuned.pth**: Fine-tuned CASDGNN model
6. **results/output_augmented/predictions.txt**: Final predictions and evaluation results
7. **bitcoin_alpha_augmented_node_mapping_to_idx.json**: Node ID to index mapping
8. **bitcoin_alpha_augmented_node_mapping_to_node.json**: Index to node ID mapping

## Expected Results

The integration should provide:
- Enhanced graph structure through balanced augmentation
- Superior prediction performance compared to SGCN alone
- Detailed evaluation metrics including accuracy, precision, recall, and F1-score
- Comprehensive logging of the entire process

## Customization

You can customize the pipeline by modifying parameters in `main.py`:

- Change dataset path in `file_path` variable
- Adjust SGCN augmentation thresholds in `thresholds` tuple
- Modify CASDGNN hyperparameters in the configuration section
- Adjust training epochs for pretraining and fine-tuning phases

## Troubleshooting

- Ensure the dataset file path is correct
- Check that all parent directory imports are working properly
- Verify CUDA availability if using GPU acceleration
- Monitor memory usage for large graphs
