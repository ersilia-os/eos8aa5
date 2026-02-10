import torch
from torch import nn

VIRTUAL_ATOM_FEATURE_PLACEHOLDER = -1
VIRTUAL_BOND_FEATURE_PLACEHOLDER = -1

class MLP(nn.Module):
    def __init__(self, d_in, d_out, n_layers, act, d_hidden=None):
        super().__init__()
        d_hidden = d_out if d_hidden is None else d_hidden
        self.in_proj = nn.Linear(d_in, d_hidden)
        self.mid = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(max(n_layers - 2, 0))])
        self.out_proj = nn.Linear(d_hidden, d_out)
        self.act = act

    def forward(self, x):
        x = self.act(self.in_proj(x))
        for layer in self.mid:
            x = self.act(layer(x))
        return self.out_proj(x)

class Residual(nn.Module):
    def __init__(self, d_in, d_out, n_ffn_layers, feat_drop, act):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.in_proj = nn.Linear(d_in, d_out)
        self.ffn = MLP(d_out, d_out, n_ffn_layers, act, d_hidden=d_out * 4)
        self.drop = nn.Dropout(feat_drop)

    def forward(self, x, y):
        x = x + self.drop(self.in_proj(y))
        y = self.norm(x)
        y = self.ffn(y)
        y = self.drop(y)
        return x + y

class AtomEmbedding(nn.Module):
    def __init__(self, d_atom, d_g, input_drop, virtual_atom_placeholder: int = -1):
        super().__init__()
        self.in_proj = nn.Linear(d_atom, d_g)
        self.virtual_atom_emb = nn.Embedding(1, d_g)
        self.drop = nn.Dropout(input_drop)
        self.virtual_atom_placeholder = int(virtual_atom_placeholder)  

    def forward(self, begin_end, vavn):
        x = self.in_proj(begin_end)
        m = (vavn == self.virtual_atom_placeholder).unsqueeze(-1).unsqueeze(-1)
        x2 = x.clone()
        x2[:, 1:2, :] = torch.where(m, self.virtual_atom_emb.weight.view(1, 1, -1), x2[:, 1:2, :])
        return torch.sum(self.drop(x2), dim=-2)


class BondEmbedding(nn.Module):
    def __init__(self, d_bond, d_g, input_drop, virtual_bond_placeholder: int = -1):
        super().__init__()
        self.in_proj = nn.Linear(d_bond, d_g)
        self.virtual_bond_emb = nn.Embedding(1, d_g)
        self.drop = nn.Dropout(input_drop)
        self.virtual_bond_placeholder = int(virtual_bond_placeholder)

    def forward(self, edge_feats, vavn):
        x = self.in_proj(edge_feats)
        m = (vavn == self.virtual_bond_placeholder).unsqueeze(-1)
        return self.drop(torch.where(m, self.virtual_bond_emb.weight.view(1, -1), x))


class TripletEmbedding(nn.Module):
    def __init__(self, d_g, d_fp, d_md, act):
        super().__init__()
        self.in_proj = MLP(d_g * 2, d_g, 2, act)
        self.fp_proj = MLP(d_fp, d_g, 2, act)
        self.md_proj = MLP(d_md, d_g, 2, act)

    def forward(self, node_h, edge_h, fp, md, vavn):
        h = self.in_proj(torch.cat([node_h, edge_h], dim=-1))
        fp_mask = (vavn == 1).unsqueeze(-1)
        md_mask = (vavn == 2).unsqueeze(-1)
        fp_h = self.fp_proj(fp).view(1, -1)
        md_h = self.md_proj(md).view(1, -1)
        h = torch.where(fp_mask, fp_h.expand_as(h), h)
        h = torch.where(md_mask, md_h.expand_as(h), h)
        return h

def _masked_mean(x, mask):
    m = mask.to(x.dtype).unsqueeze(-1)
    s = (x * m).sum(dim=0, keepdim=True)
    c = m.sum(dim=0, keepdim=True).clamp_min(1.0)
    return s / c

class TripletTransformerIncoming(nn.Module):
    def __init__(self, d_feats, n_heads, n_ffn_dense_layers, feat_drop, attn_drop, act):
        super().__init__()
        self.d = d_feats
        self.h = n_heads
        self.dh = d_feats // n_heads
        self.scale = d_feats ** (-0.5)

        self.attention_norm = nn.LayerNorm(d_feats)
        self.qkv = nn.Linear(d_feats, d_feats * 3)
        self.node_out_layer = Residual(d_feats, d_feats, n_ffn_dense_layers, feat_drop, act)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, triplet_h, mask_nodes, src, dst, edge_bias, mask_edges, inc_idx, inc_mask):

        H = self.h
        dh = self.dh

        x = self.attention_norm(triplet_h)

        qkv = self.qkv(x).reshape(-1, 3, H, dh)
        q = qkv[:, 0, :, :] * self.scale 
        k = qkv[:, 1, :, :]              
        v = qkv[:, 2, :, :]               

        se_take = torch.where(mask_edges, src, torch.zeros_like(src))
        de_take = torch.where(mask_edges, dst, torch.zeros_like(dst))

        dot_all = (q[se_take] * k[de_take]).sum(-1)
        dot_all = torch.where(mask_edges.unsqueeze(-1), dot_all, torch.zeros_like(dot_all))

        edge_bias = torch.where(mask_edges.unsqueeze(-1), edge_bias, torch.zeros_like(edge_bias))

        scores_all = dot_all + edge_bias  

        v_src = v[se_take]
        v_src = torch.where(mask_edges.unsqueeze(-1).unsqueeze(-1), v_src, torch.zeros_like(v_src))

        idx = inc_idx.clamp_min(0)      
        s_in = scores_all[idx]          
        v_in = v_src[idx]               

        m_in = inc_mask & mask_nodes.unsqueeze(-1)   
        s_in = torch.where(m_in.unsqueeze(-1), s_in, torch.full_like(s_in, -1e9))

        a_in = torch.softmax(s_in, dim=1)           
        a_in = self.attn_drop(a_in)

        out = (v_in * a_in.unsqueeze(-1)).sum(dim=1) 
        out = out.reshape(-1, H * dh)               

        return self.node_out_layer(triplet_h, out)


class LiGhTIncoming(nn.Module):
    def __init__(self, d_g, d_hpath_ratio, path_length, n_mol_layers, n_heads, n_ffn_dense_layers, feat_drop, attn_drop, act):
        super().__init__()
        self.path_length = path_length
        self.d_g = d_g
        self.h = n_heads
        self.d_trip = d_g // d_hpath_ratio

        self.mask_emb = nn.Embedding(1, d_g)
        self.path_len_emb = nn.Embedding(path_length + 1, d_g)
        self.virtual_path_emb = nn.Embedding(1, d_g)
        self.self_loop_emb = nn.Embedding(1, d_g)

        self.dist_attn_layer = nn.Sequential(nn.Linear(d_g, d_g), act, nn.Linear(d_g, n_heads))
        self.trip_fortrans = nn.ModuleList([MLP(d_g, self.d_trip, 2, act) for _ in range(path_length)])
        self.path_attn_layer = nn.Sequential(nn.Linear(self.d_trip, self.d_trip), act, nn.Linear(self.d_trip, n_heads))

        self.mol_T_layers = nn.ModuleList([
            TripletTransformerIncoming(d_g, n_heads, n_ffn_dense_layers, feat_drop, attn_drop, act)
            for _ in range(n_mol_layers)
        ])

    def _featurize_path(self, path, vp, sl):
        mask = (path >= 0).to(torch.int64)
        plen = mask.sum(dim=-1).clamp_min(0)
        dist_h = self.path_len_emb(plen)
        dist_h = torch.where(vp.unsqueeze(-1), self.virtual_path_emb.weight.view(1, -1), dist_h)
        dist_h = torch.where(sl.unsqueeze(-1), self.self_loop_emb.weight.view(1, -1), dist_h)
        return dist_h

    def _init_path(self, triplet_h, path):
        p = path.clone()
        p[p < -99] = -1

        proj_list = [self.trip_fortrans[i](triplet_h) for i in range(self.path_length)]
        z = torch.zeros((1, self.d_trip), device=triplet_h.device, dtype=triplet_h.dtype)

        N = triplet_h.shape[0]  

        xs = []
        for i in range(self.path_length):
            proj = proj_list[i]                 
            proj2 = torch.cat([proj, z], dim=0)

            pi = p[:, i]
            idx = torch.where(pi >= 0, pi, torch.full_like(pi, N))
            xs.append(proj2[idx])              

        x = torch.stack(xs, dim=-1)          
        m = (p >= 0).to(triplet_h.dtype)       
        denom = m.sum(dim=-1, keepdim=True).clamp_min(1.0)

        return (x * m.unsqueeze(1)).sum(dim=-1) / denom

    def forward(self, triplet_h, mask_nodes, src, dst, path, vp, sl, mask_edges, inc_idx, inc_mask):
        dist_h = self._featurize_path(path, vp, sl)
        path_h = self._init_path(triplet_h, path)
        dist_attn = self.dist_attn_layer(dist_h)
        path_attn = self.path_attn_layer(path_h)
        edge_bias = dist_attn + path_attn
        for layer in self.mol_T_layers:
            triplet_h = layer(triplet_h, mask_nodes, src, dst, edge_bias, mask_edges, inc_idx, inc_mask)
        return triplet_h

class LiGhTPredictorTensor(nn.Module):
    def __init__(
        self,
        d_node_feats,
        d_edge_feats,
        d_g_feats,
        d_fp_feats,
        d_md_feats,
        d_hpath_ratio,
        n_mol_layers,
        path_length,
        n_heads,
        n_ffn_dense_layers,
        input_drop,
        feat_drop,
        attn_drop,
        readout_mode="mean",
    ):
        super().__init__()
        act = nn.GELU()
        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb = TripletEmbedding(d_g_feats, d_fp_feats, d_md_feats, act)
        self.model = LiGhTIncoming(d_g_feats, d_hpath_ratio, path_length, n_mol_layers, n_heads, n_ffn_dense_layers, feat_drop, attn_drop, act)
        self.readout_mode = readout_mode

    def forward(self, begin_end, edge, vavn, mask_nodes, src, dst, path, vp, sl, mask_edges, inc_idx, inc_mask, fp, md):
        node_h = self.node_emb(begin_end, vavn)
        edge_h = self.edge_emb(edge, vavn)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, vavn)
        triplet_h = self.model(triplet_h, mask_nodes, src, dst, path, vp, sl, mask_edges, inc_idx, inc_mask)

        fp_vn = _masked_mean(triplet_h, (vavn == 1) & mask_nodes)
        md_vn = _masked_mean(triplet_h, (vavn == 2) & mask_nodes)
        readout = _masked_mean(triplet_h, (vavn < 1) & mask_nodes)

        return torch.cat([fp_vn, md_vn, readout], dim=-1).squeeze(0)
