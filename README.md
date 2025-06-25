# CASDGNN: Centrality-Aware Signed Directed Graph Neural Network

CASDGNN is a sophisticated graph neural network architecture designed for signed directed networks, incorporating node centrality information to enhance the learning of node representations and improve link sign prediction.

## Model Architecture

CASDGNN consists of several key components:

1. **Centrality-Aware Encoder**: Combines node features with network centrality metrics (betweenness and closeness centrality) to create richer node representations.

2. **Signed Directed Attention**: A multi-head attention mechanism that processes four types of relations (positive/negative × incoming/outgoing) while considering node sign influence.

3. **Graph Transformer**: Multiple layers of attention mechanisms for capturing complex node interactions.

4. **Link Sign Prediction**: Predicts the sign of edges using node embeddings and focal loss for handling class imbalance.

## File Structure and Components

- `model.py`: Contains the core model architecture including CentralityAwareEncoder, SignedDirectedAttention, GraphTransformer, and CA_SDGNN classes
- `data_utils.py`: Functions for data loading, preprocessing, and creating adjacency matrices
- `train_utils.py`: Training utilities for pretraining and fine-tuning
- `eval_utils.py`: Evaluation and inference functions
- `main.py`: Main script for running the model
- `fea_extra.py`: Feature extraction utilities
- `common.py`: Common utilities and helper functions

## Installation

1. Create a virtual environment:
```bash
python -m venv capstone
```

2. Activate the virtual environment:
   - Windows:
   ```bash
   .\capstone\Scripts\activate
   ```
   - Linux/Mac:
   ```bash
   source capstone/bin/activate
   ```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

The model training process consists of three main steps:

### 1. Pretraining

This step learns initial node representations through self-supervised learning:

```bash
python main.py \
  --dataset bitcoin_alpha \
  --mode pretrain \
  --train_file experiment-data/bitcoin_alpha-train-1.edgelist \
  --device cuda \
  --epochs 500 \
  --node_feat_dim 16 \
  --embed_dim 16 \
  --num_heads 4 \
  --num_layers 2 \
  --dropout_rate 0.1 \
  --pretrain_path embeddings/bitcoin_alpha_ca_sdgnn_pretrained.pth
```

### 2. Fine-tuning

Fine-tune the model for the link sign prediction task:

```bash
python main.py \
  --dataset bitcoin_alpha \
  --mode finetune \
  --train_file experiment-data/bitcoin_alpha-train-1.edgelist \
  --device cuda \
  --epochs 50 \
  --node_feat_dim 16 \
  --embed_dim 16 \
  --num_heads 4 \
  --num_layers 2 \
  --dropout_rate 0.1 \
  --lr 0.0005 \
  --weight_decay 0.0001 \
  --pretrain_path embeddings/bitcoin_alpha_ca_sdgnn_pretrained.pth \
  --finetune_path embeddings/bitcoin_alpha_ca_sdgnn_finetuned.pth
```

### 3. Inference

Run inference on test data:

```bash
python main.py \
  --dataset bitcoin_alpha \
  --mode infer \
  --test_file experiment-data/bitcoin_alpha-test-1.edgelist \
  --device cuda \
  --node_feat_dim 16 \
  --embed_dim 16 \
  --num_heads 4 \
  --num_layers 2 \
  --dropout_rate 0.1 \
  --pretrain_path embeddings/bitcoin_alpha_ca_sdgnn_pretrained.pth \
  --finetune_path embeddings/bitcoin_alpha_ca_sdgnn_finetuned.pth \
  --output_dir output
```

## Model Parameters

- `node_feat_dim`: Dimension of initial node features
- `embed_dim`: Dimension of node embeddings
- `num_heads`: Number of attention heads
- `num_layers`: Number of transformer layers
- `dropout_rate`: Dropout rate for regularization
- `lr`: Learning rate for optimization
- `weight_decay`: L2 regularization parameter

## Advanced Usage: DropEdge Augmentation Integration

The project includes an advanced integration that combines CASDGNN with graph augmentation techniques. This hybrid approach uses SGCN for graph augmentation and CASDGNN for final predictions.

### Running the Integration

```bash
cd dropedge_augmentation
python main.py
```

This integration provides:
- Enhanced graph structure through balanced edge augmentation
- Superior prediction performance on augmented graphs
- Comprehensive evaluation metrics and detailed logging

For more details, see the `dropedge_augmentation/readme.md` file.