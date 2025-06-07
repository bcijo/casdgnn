import networkx as nx
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from select_candidates import select_beneficial_candidates

# Example Usage

# 1. Create a sample graph G
G = nx.Graph()
G.add_edge(0, 1, sign=1)
G.add_edge(1, 2, sign=1)
G.add_edge(2, 0, sign=1)  # Balanced triangle (0,1,2)
G.add_edge(2, 3, sign=-1)
G.add_edge(3, 4, sign=1)

print("Original Graph Edges:")
for u, v, data in G.edges(data=True):
    print(f"({u}, {v}, sign={data['sign']})")
print("-----")

# 2. Define add_candidates and del_candidates
add_candidates = [
    (0, 4, 1),  # Try to add edge (0,4) with sign 1. Potential triangle (0,4,3) with (3,4,s=1), (0,3,s=?) - no (0,3)
                # Potential triangle (0,4,2) with (2,4,s=?) - no (2,4)
                # Should be safe.
    (1, 3, -1) # Try to add edge (1,3) with sign -1. Potential triangle (1,3,2) with (1,2,s=1), (2,3,s=-1)
                # Product: (-1)*(1)*(-1) = 1 > 0. So, (1,3,-1) should be safe.
]
del_candidates = [
    (2, 0, 1)   # Delete edge (2,0) with sign 1
]

print("Add Candidates:", add_candidates)
print("Delete Candidates:", del_candidates)
print("-----")

# 3. Call the function
G_aug, filtered_add_candidates = select_beneficial_candidates(G, add_candidates, del_candidates)

# 4. Print the output
print("Augmented Graph Edges (G_aug):")
for u, v, data in G_aug.edges(data=True):
    print(f"({u}, {v}, sign={data['sign']})")
print("-----")

print("Filtered Add Candidates:")
print(filtered_add_candidates)
print("-----")

# You can add more specific checks here if needed, for example:
# Check if (2,0) was actually deleted
if not G_aug.has_edge(2,0):
    print("Edge (2,0) successfully deleted.")
else:
    print("Error: Edge (2,0) was NOT deleted.")

# Check if (0,4,1) was added
if G_aug.has_edge(0,4) and G_aug[0][4].get('sign') == 1:
    print("Edge (0,4, sign=1) successfully added.")
else:
    print("Error: Edge (0,4, sign=1) was NOT added or sign is incorrect.")

# Check if (1,3,-1) was added
if G_aug.has_edge(1,3) and G_aug[1][3].get('sign') == -1:
    print("Edge (1,3, sign=-1) successfully added.")
else:
    print("Error: Edge (1,3, sign=-1) was NOT added or sign is incorrect.")


# Example of an edge that should NOT be added because it creates an unbalanced triangle
print("\nTesting an unbeneficial addition:")
G_test_unbalanced = nx.Graph()
G_test_unbalanced.add_edge(5, 6, sign=1)
G_test_unbalanced.add_edge(6, 7, sign=1)
# If we add (5,7) with sign -1, triangle (5,6,7) becomes 1*1*(-1) = -1 (unbalanced)
add_unbeneficial = [(5, 7, -1)]
del_none = []

G_aug_un, filtered_add_un = select_beneficial_candidates(G_test_unbalanced, add_unbeneficial, del_none)

print("Original graph for unbeneficial test:")
for u,v,data in G_test_unbalanced.edges(data=True): print(f"({u},{v}, sign={data['sign']})")

print("Augmented graph after unbeneficial test (should be same as original):")
for u,v,data in G_aug_un.edges(data=True): print(f"({u},{v}, sign={data['sign']})")

print("Filtered add candidates for unbeneficial test (should be empty):")
print(filtered_add_un)

if not G_aug_un.has_edge(5,7):
    print("Edge (5,7, sign=-1) was correctly NOT added as it's unbeneficial.")
else:
    print("Error: Edge (5,7, sign=-1) was added but it's unbeneficial.")
