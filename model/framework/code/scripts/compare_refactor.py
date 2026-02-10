import os
import numpy as np
import torch
import sys
sys.path.append("..")
from src.model_config import config_dict
from src.model.light import LiGhTPredictor
from src.model.light_tensor import LiGhTPredictorTensor
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from scripts.preprocess_downstream_dataset import extract_features_from_smiles

CONFIG_NAME = "base"
CONFIG = config_dict[CONFIG_NAME]
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CKPT = os.path.join(project_root, "checkpoints", "pretrained", CONFIG_NAME, f"{CONFIG_NAME}.pth")

def remap_key(k):
    k = k.replace("module.", "")
    k = k.replace("edge_emb.virutal_bond_emb.", "edge_emb.virtual_bond_emb.")
    return k

smiles = ["CCO"]

feat = extract_features_from_smiles(smiles, n_jobs=1, path_length=CONFIG["path_length"])
g = feat["graphs"][0]
fp = torch.tensor(feat["ecfp"][0], dtype=torch.float32)
md = torch.tensor(feat["descriptors"][0], dtype=torch.float32)

vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)

orig = LiGhTPredictor(
    d_node_feats=CONFIG["d_node_feats"],
    d_edge_feats=CONFIG["d_edge_feats"],
    d_g_feats=CONFIG["d_g_feats"],
    d_hpath_ratio=CONFIG["d_hpath_ratio"],
    n_mol_layers=CONFIG["n_mol_layers"],
    path_length=CONFIG["path_length"],
    n_heads=CONFIG["n_heads"],
    n_ffn_dense_layers=CONFIG["n_ffn_dense_layers"],
    input_drop=0,
    attn_drop=0,
    feat_drop=0,
    n_node_types=vocab.vocab_size,
).eval()

ref = LiGhTPredictorTensor(
    d_node_feats=CONFIG["d_node_feats"],
    d_edge_feats=CONFIG["d_edge_feats"],
    d_g_feats=CONFIG["d_g_feats"],
    d_fp_feats=512,
    d_md_feats=int(md.shape[0]),
    d_hpath_ratio=CONFIG["d_hpath_ratio"],
    n_mol_layers=CONFIG["n_mol_layers"],
    path_length=CONFIG["path_length"],
    n_heads=CONFIG["n_heads"],
    n_ffn_dense_layers=CONFIG["n_ffn_dense_layers"],
    input_drop=0,
    feat_drop=0,
    attn_drop=0,
).eval()

state = torch.load(CKPT, map_location="cpu")
orig.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
ref.load_state_dict({remap_key(k): v for k, v in state.items()}, strict=False)
print(g)
with torch.no_grad():
    a = orig.generate_fps(g, fp.unsqueeze(0), md.unsqueeze(0))[0].cpu().numpy()

    begin_end = g.ndata["begin_end"]
    edge = g.ndata["edge"]
    vavn = g.ndata["vavn"]
    mask_nodes = torch.ones((g.num_nodes(),), dtype=torch.bool)

    src, dst = g.edges()
    src = src.long()
    dst = dst.long()

    path = g.edata["path"].long()
    vp = g.edata["vp"].bool()
    sl = g.edata["sl"].bool()
    mask_edges = torch.ones((g.num_edges(),), dtype=torch.bool)

    N = g.num_nodes()
    E = g.num_edges()
    K = E

    inc_idx = torch.full((N, K), -1, dtype=torch.int64)
    inc_mask = torch.zeros((N, K), dtype=torch.bool)
    pos = torch.zeros((N,), dtype=torch.int64)
    for e in range(E):
        d = int(dst[e])
        p = int(pos[d])
        inc_idx[d, p] = e
        inc_mask[d, p] = True
        pos[d] += 1

    b = ref(
        begin_end, edge, vavn, mask_nodes,
        src, dst, path, vp, sl, mask_edges,
        inc_idx, inc_mask,
        fp, md
    ).cpu().numpy()

print("max_abs", float(np.max(np.abs(a - b))))
print("mean_abs", float(np.mean(np.abs(a - b))))
print("cos", float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
