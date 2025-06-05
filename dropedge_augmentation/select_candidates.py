import networkx as nx

def is_balanced_triangle(G, u, v, w):
    # Check if triangle (u, v, w) is balanced (product of edge signs > 0)
    try:
        sign_uv = G[u][v]['sign']
        sign_vw = G[v][w]['sign']
        sign_wu = G[w][u]['sign']
        return sign_uv * sign_vw * sign_wu > 0
    except KeyError:
        return True  # If any edge doesn't exist, consider it balanced (no triangle formed)

def select_beneficial_candidates(G, add_candidates, del_candidates):
    G_aug = G.copy()
    
    # Apply deletions (always beneficial as they don't create new triangles)
    for i, j, sign in del_candidates:
        if G_aug.has_edge(i, j):
            G_aug.remove_edge(i, j)
    
    # Filter additions to avoid new unbalanced triangles
    filtered_add_candidates = []
    for i, j, sign in add_candidates:
        G_temp = G_aug.copy()
        G_temp.add_edge(i, j, sign=sign)
        is_safe = True
        
        # Check all triangles involving the new edge
        for k in G_temp.nodes():
            if k != i and k != j:
                if G_temp.has_edge(i, k) and G_temp.has_edge(k, j):
                    if not is_balanced_triangle(G_temp, i, j, k):
                        is_safe = False
                        break
        
        if is_safe:
            filtered_add_candidates.append((i, j, sign))
            G_aug.add_edge(i, j, sign=sign)
    
    return G_aug, filtered_add_candidates

# Example usage
G_aug, filtered_add_candidates = select_beneficial_candidates(G, add_candidates, del_candidates)