import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CentralityAwareEncoder(nn.Module):
    def __init__(self, node_feat_dim, embed_dim, centrality_dim, device):
        super(CentralityAwareEncoder, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.embed_dim = embed_dim
        self.centrality_dim = centrality_dim
        self.device = device
        self.feature_combiner = nn.Linear(node_feat_dim, embed_dim)
        self.centrality_encoder = nn.Linear(centrality_dim, embed_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.feature_combiner.weight)
        self.feature_combiner.bias.data.fill_(0.01)
        nn.init.kaiming_normal_(self.centrality_encoder.weight)
        self.centrality_encoder.bias.data.fill_(0.01)

    def forward(self, nodes, node_features, centrality_features):
        nodes_tensor = nodes if isinstance(nodes, torch.Tensor) else torch.LongTensor(nodes).to(self.device)
        self_feats = node_features[nodes_tensor]
        betweenness = centrality_features["betweenness"][nodes]
        closeness = centrality_features["closeness"][nodes]
        central_feats = torch.stack([betweenness, closeness], dim=1).to(self.device)
        central_feats_encoded = self.centrality_encoder(central_feats)
        combined_feats = self.feature_combiner(self_feats) + central_feats_encoded
        return combined_feats

class SignedDirectedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_relations=4, dropout_rate=0.1):
        super(SignedDirectedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.dropout = nn.Dropout(dropout_rate)
        self.sqrt_d = math.sqrt(embed_dim)
        self.query_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim * num_heads) for _ in range(num_relations)])
        self.key_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim * num_heads) for _ in range(num_relations)])
        self.value_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim * num_heads) for _ in range(num_relations)])
        self.attn_weight_proj = nn.Linear(embed_dim * num_heads * num_relations, embed_dim)
        self.sign_weight = nn.Parameter(torch.Tensor(num_heads, num_relations))
        nn.init.xavier_uniform_(self.sign_weight)

    def forward(self, node_embeddings, node_sign_influence, adj_matrices):
        num_nodes = node_embeddings.size(0)
        outputs = []
        
        for r in range(self.num_relations):
            Q = self.query_layers[r](node_embeddings).view(num_nodes, self.num_heads, self.embed_dim)
            K = self.key_layers[r](node_embeddings).view(num_nodes, self.num_heads, self.embed_dim)
            V = self.value_layers[r](node_embeddings).view(num_nodes, self.num_heads, self.embed_dim)
            adj_matrix = adj_matrices[r].to_dense()
            edge_index = (adj_matrix > 0).nonzero(as_tuple=False)
            src, tgt = edge_index[:, 0], edge_index[:, 1]
            
            Q_edges = Q[src]
            K_edges = K[tgt]
            scores = (Q_edges * K_edges).sum(dim=-1) / self.sqrt_d
            sign_factor = node_sign_influence[src].unsqueeze(1) * self.sign_weight[:, r].unsqueeze(0)
            scores = scores * sign_factor
            
            attention_weights = torch.empty_like(scores)
            for h in range(self.num_heads):
                scores_h = scores[:, h]
                unique_src, inverse_indices = torch.unique(src, return_inverse=True)
                max_per_src = torch.zeros(unique_src.size(0), device=scores_h.device).scatter_reduce(0, inverse_indices, scores_h, reduce="amax", include_self=False)
                exp_scores = torch.exp(scores_h - max_per_src[inverse_indices])
                sum_exp = torch.zeros(unique_src.size(0), device=scores_h.device).scatter_add_(0, inverse_indices, exp_scores)
                attention_weights[:, h] = exp_scores / (sum_exp[inverse_indices] + 1e-10)
            
            attention_weights = self.dropout(attention_weights)
            V_edges = V[tgt]
            weighted_V = V_edges * attention_weights.unsqueeze(-1)
            out_per_head = torch.zeros(self.num_heads, num_nodes, self.embed_dim, device=node_embeddings.device)
            for h in range(self.num_heads):
                out_per_head[h].index_add_(0, src, weighted_V[:, h, :])
            out = out_per_head.transpose(0, 1).reshape(num_nodes, -1)
            outputs.append(out)
        
        combined_out = torch.cat(outputs, dim=-1)
        final_out = self.attn_weight_proj(combined_out)
        return final_out

class GraphTransformer(nn.Module):
    def __init__(self, node_feat_dim, embed_dim, centrality_dim, num_heads, num_layers, device, dropout_rate=0.1):
        super(GraphTransformer, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.encoder = CentralityAwareEncoder(node_feat_dim, embed_dim, centrality_dim, device)
        self.transformer_layers = nn.ModuleList([
            SignedDirectedAttention(embed_dim, num_heads, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, nodes, node_features, centrality_features, adj_matrices, node_sign_influence):
        x = self.encoder(nodes, node_features, centrality_features)
        for i in range(self.num_layers):
            attn_out = self.transformer_layers[i](x, node_sign_influence, adj_matrices)
            x = self.layer_norms[i](x + attn_out)
            x = self.dropout(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

class CA_SDGNN(nn.Module):
    def __init__(self, node_feat_dim, embed_dim, centrality_dim, num_heads, num_layers, device, dropout_rate=0.1, use_focal_loss=True):
        super(CA_SDGNN, self).__init__()
        self.transformer = GraphTransformer(
            node_feat_dim, embed_dim, centrality_dim, num_heads, num_layers, device, dropout_rate
        )
        self.linear = nn.Linear(embed_dim * 2, 1)
        self.score_function1 = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())
        self.score_function2 = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())
        self.fc = nn.Linear(embed_dim * 2, 1)
        self.use_focal_loss = use_focal_loss
        self.focal_loss = FocalLoss(alpha=2, gamma=2) if use_focal_loss else None
        self.device = device

    def _store_features(self, node_features, centralityFeatures, adj_matrices, node_sign_influence):
        # Store features for use in criterion
        self._features = {
            'node_features': node_features,
            'centrality_features': centralityFeatures,
            'adj_matrices': adj_matrices,
            'node_sign_influence': node_sign_influence
        }

    def _get_stored_features(self):
        if not hasattr(self, '_features'):
            raise RuntimeError("Features not stored. Call forward() before criterion()")
        return self._features

    def criterion(self, nodes, pos_out_neighbors, neg_out_neighbors, pos_in_neighbors, neg_in_neighbors, edge_list, labels, weight_dict):
        features = self._get_stored_features()
        
        # Get embeddings for all nodes first to avoid duplicate calculation
        unique_nodes = set(nodes)
        for u, v in edge_list:
            unique_nodes.add(u)
            unique_nodes.add(v)
        unique_nodes_list = list(unique_nodes)
        unique_nodes_dict = {n: i for i, n in enumerate(unique_nodes_list)}
        
        # Get embeddings using stored features
        nodes_embs = self.forward(
            unique_nodes_list,
            features['node_features'],
            features['centrality_features'],
            features['adj_matrices'],
            features['node_sign_influence']
        )
        
        loss_total = 0
        for index, node in enumerate(nodes):
            z1 = nodes_embs[unique_nodes_dict[node], :]
            pos_out_neigs = [unique_nodes_dict[i] for i in pos_out_neighbors.get(node, []) if i in unique_nodes_dict]
            neg_out_neigs = [unique_nodes_dict[i] for i in neg_out_neighbors.get(node, []) if i in unique_nodes_dict]
            pos_in_neigs = [unique_nodes_dict[i] for i in pos_in_neighbors.get(node, []) if i in unique_nodes_dict]
            neg_in_neigs = [unique_nodes_dict[i] for i in neg_in_neighbors.get(node, []) if i in unique_nodes_dict]
            
            for neigs, sign, direction in [
                (pos_out_neigs, 1, "out"),
                (neg_out_neigs, 0, "out"),
                (pos_in_neigs, 1, "in"),
                (neg_in_neigs, 0, "in")
            ]:
                if len(neigs) > 0:
                    neig_embs = nodes_embs[neigs, :]
                    targets = torch.ones(len(neigs)).to(nodes_embs.device) if sign == 1 else torch.zeros(len(neigs)).to(nodes_embs.device)
                    logits = torch.einsum("nj,j->n", [neig_embs, z1])
                    
                    if self.use_focal_loss and self.focal_loss:
                        loss_pku = self.focal_loss(logits, targets)
                    else:
                        loss_pku = F.binary_cross_entropy_with_logits(logits, targets)
                    
                    if direction == "out":
                        neigs_nodes = [unique_nodes_list[n] for n in neigs]
                        neigs_weights = torch.FloatTensor([weight_dict.get(node, {}).get(n, 1.0) for n in neigs_nodes]).to(nodes_embs.device)
                        z11 = z1.repeat(len(neigs), 1)
                        rs = self.fc(torch.cat([z11, neig_embs], 1)).squeeze(-1)
                        
                        if self.use_focal_loss and self.focal_loss:
                            loss_pku += self.focal_loss(rs, targets)
                        else:
                            loss_pku += F.binary_cross_entropy_with_logits(rs, targets, weight=neigs_weights)
                        
                        s1 = self.score_function1(z1).repeat(len(neigs), 1)
                        s2 = self.score_function2(neig_embs)
                        threshold = 0.5 if sign == 0 else -0.5
                        q = torch.where(
                            (s1 - s2) > threshold,
                            torch.Tensor([threshold]).repeat(s1.shape).to(nodes_embs.device),
                            s1 - s2
                        )
                        tmp = (q - (s1 - s2))
                        loss_pku += 5 * torch.einsum("ij,ij->", [tmp, tmp])
                    
                    loss_total += loss_pku
        
        # Link prediction loss
        if edge_list and labels is not None:
            edge_logits = self.forward(nodes, features['node_features'], features['centrality_features'],
                                    features['adj_matrices'], features['node_sign_influence'], edge_list=edge_list)
            if self.use_focal_loss and self.focal_loss:
                link_loss = self.focal_loss(edge_logits, labels)
            else:
                link_loss = F.binary_cross_entropy_with_logits(edge_logits, labels)
            loss_total += link_loss
        
        return loss_total

    def forward(self, nodes, node_features, centrality_features, adj_matrices, node_sign_influence, edge_list=None):
        # Store features for use in criterion
        self._store_features(node_features, centrality_features, adj_matrices, node_sign_influence)
        
        node_embeddings = self.transformer(nodes, node_features, centrality_features, adj_matrices, node_sign_influence)
        
        if edge_list is not None:
            edge_embeddings = []
            for edge in edge_list:
                source_emb = node_embeddings[edge[0]]
                target_emb = node_embeddings[edge[1]]
                combined_emb = torch.cat([source_emb, target_emb], dim=0)
                edge_embeddings.append(combined_emb)
            edge_embeddings = torch.stack(edge_embeddings).to(node_embeddings.device)
            link_logits = self.linear(edge_embeddings)
            return link_logits.squeeze()
        
        return node_embeddings