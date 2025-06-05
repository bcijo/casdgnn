import networkx as nx
import torch

def compute_edge_difficulty(G, edge_index):
    # Find all triangles
    triangles = []
    for u in G.nodes():
        for v, w in combinations(G.neighbors(u), 2):
            if G.has_edge(v, w) or G.has_edge(w, v):
                triangles.append((u, v, w))
    
    # Compute local balance degree and difficulty score for each edge
    difficulty_scores = {}
    for i, j in edge_index.t().tolist():
        edge = (i, j)
        pos_triangles, neg_triangles = 0, 0
        for u, v, w in triangles:
            if (i in (u, v, w) and j in (u, v, w)):
                if is_balanced_triangle(G, u, v, w):
                    pos_triangles += 1
                else:
                    neg_triangles += 1
        total_triangles = pos_triangles + neg_triangles
        if total_triangles == 0:
            local_balance = 0  # Default if no triangles
        else:
            local_balance = (pos_triangles - neg_triangles) / total_triangles
        difficulty_scores[edge] = (1 - local_balance) / 2
    
    return difficulty_scores

def curriculum_training(model, edge_index, edge_signs, difficulty_scores, pos_edge_index, neg_edge_index, epochs=300, T=100, lambda_0=0.2):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    
    # Sort edges by difficulty
    sorted_edges = sorted(difficulty_scores.items(), key=lambda x: x[1])
    edge_order = [(edge[0], edge[1]) for edge, _ in sorted_edges]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(pos_edge_index, neg_edge_index)
        
        # Pacing function
        g_t = min(1, lambda_0 + (1 - lambda_0) * epoch / T)
        num_edges = int(len(edge_order) * g_t)
        selected_edges = edge_order[:num_edges]
        
        # Create mask for selected edges
        mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        for i, j in selected_edges:
            mask[(edge_index[0] == i) & (edge_index[1] == j)] = True
        
        # Compute loss on selected edges
        logits = model.predict_edge(z, edge_index[:, mask])
        loss = F.cross_entropy(logits, (edge_signs[mask] + 1).long())
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Edges used: {num_edges}')
    
    return model

# Example usage
difficulty_scores = compute_edge_difficulty(G_aug, edge_index)
sgcn = curriculum_training(sgcn, edge_index, edge_signs, difficulty_scores, pos_edge_index, neg_edge_index)