# DropEdge Augmentation for Signed Graph Networks

This directory contains the implementation of a DropEdge-inspired augmentation technique tailored for signed graph networks. The primary goal is to improve the performance of Signed Graph Convolutional Networks (SGCNs) by strategically adding or removing edges to enhance the structural properties of the graph, particularly focusing on the concept of structural balance.

The process involves:
1.  Training an initial SGCN model on the original signed graph.
2.  Using the trained model to predict potential edge additions and deletions.
3.  Filtering these candidate edges to ensure that modifications (especially additions) do not introduce new unbalanced triangles, thereby preserving or improving the structural balance of the graph.
4.  Retraining the SGCN model on the augmented graph, potentially using a curriculum learning strategy where the model learns from "easier" edges first.

## File Descriptions

*   **`main.py`**: The main script that orchestrates the entire workflow. It handles dataset loading, initial model training, candidate generation, candidate selection, graph augmentation, and final model training (potentially with curriculum learning).
*   **`load_dataset.py`**: Contains functions to load signed graph datasets (e.g., Bitcoin Alpha, Bitcoin OTC) from CSV files. It processes the data into a NetworkX graph object and prepares PyTorch tensors for edge indices and signs.
*   **`sgcn.py`**: Defines the Signed Graph Convolutional Network (SGCN) model architecture. This includes the `SGCNConv` layer, which handles message passing for positive and negative edges separately, and the overall `SGCN` model. It also includes training and evaluation helper functions for the SGCN.
*   **`generate_candidates.py`**: This script uses a trained SGCN model to predict the likelihood of potential edges (and existing ones). Based on these predictions and specified thresholds, it generates a list of candidate edges for addition and deletion.
*   **`select_candidates.py`**: Implements the logic for filtering the candidate edges. It prioritizes deletions and then selectively adds edges, ensuring that new additions do not create unbalanced triangles in the graph.
*   **`edge_difficulty.py`**: This script defines functions to compute an "edge difficulty" score, often based on local balance properties. It also implements a curriculum training approach, where the SGCN model is trained by gradually introducing edges from easiest to hardest.
*   **`bitcoin_alpha.csv`**: An example dataset (Bitcoin Alpha) used for the signed graph analysis.
*   **`bitcoin-otc-augmented.edgelist`**: An example of an output file where the augmented graph's edgelist might be saved.

## Running the Code

To run the main augmentation process, navigate to this directory in your terminal and execute:

```bash
cd casdgnn/dropedge_augmentation
```

```bash
python main.py
```

**Prerequisites:**
*   Python 3.10.9
*   PyTorch
*   NetworkX
*   Pandas
*   NumPy

Ensure all dependencies are installed. You might need to adjust the dataset path within `main.py` (currently hardcoded for `bitcoin_alpha.csv`) if you are using a different dataset or if your file is in a different location.

The script will:
1.  Load the specified dataset.
2.  Train an initial SGCN model.
3.  Generate candidate edges for addition/deletion.
4.  Select beneficial candidates to create an augmented graph.
5.  Train the SGCN model on this augmented graph using curriculum learning.
6.  Optionally, save the augmented graph to an edgelist file.
