import os
import sys
import torch

sys.path.append("..")

from src.model_config import config_dict
from src.model.light_tensor import LiGhTPredictorTensor


CONFIG_NAME = "base"
CFG = config_dict[CONFIG_NAME]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "pretrained", CONFIG_NAME, f"{CONFIG_NAME}.pth")
TS_OUT = os.path.join(PROJECT_ROOT, "checkpoints", "pretrained", CONFIG_NAME, "light.ts.pt")

D_FP = 512
D_MD = 200


def remap_key(k: str) -> str:
    k = k.replace("module.", "")
    k = k.replace("edge_emb.virutal_bond_emb.", "edge_emb.virtual_bond_emb.")
    return k


def main():
    torch.set_grad_enabled(False)

    model = LiGhTPredictorTensor(
        d_node_feats=CFG["d_node_feats"],
        d_edge_feats=CFG["d_edge_feats"],
        d_g_feats=CFG["d_g_feats"],
        d_fp_feats=D_FP,
        d_md_feats=D_MD,
        d_hpath_ratio=CFG["d_hpath_ratio"],
        n_mol_layers=CFG["n_mol_layers"],
        path_length=CFG["path_length"],
        n_heads=CFG["n_heads"],
        n_ffn_dense_layers=CFG["n_ffn_dense_layers"],
        input_drop=0.0,
        feat_drop=0.0,
        attn_drop=0.0,
    ).eval()

    state = torch.load(CKPT, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {remap_key(k): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Missing:", len(missing))
    if missing:
        print("  e.g.", missing[:10])
    print("Unexpected:", len(unexpected))
    if unexpected:
        print("  e.g.", unexpected[:10])

    try:
        ts = torch.jit.script(model)
        ts.save(TS_OUT)
        print("Saved TorchScript (script):", TS_OUT)
        return
    except Exception as e:
        print("torch.jit.script failed, falling back to trace.")
        print("Reason:", repr(e))

    N = 4
    E = 14
    K = E 

    begin_end = torch.zeros((N, 2, CFG["d_node_feats"]), dtype=torch.float32)
    edge = torch.zeros((N, CFG["d_edge_feats"]), dtype=torch.float32)
    vavn = torch.zeros((N,), dtype=torch.int64)
    mask_nodes = torch.ones((N,), dtype=torch.bool)

    src = torch.zeros((E,), dtype=torch.int64)
    dst = torch.zeros((E,), dtype=torch.int64)
    path = torch.full((E, CFG["path_length"]), -1, dtype=torch.int64)
    vp = torch.zeros((E,), dtype=torch.bool)
    sl = torch.zeros((E,), dtype=torch.bool)
    mask_edges = torch.ones((E,), dtype=torch.bool)

    inc_idx = torch.full((N, K), -1, dtype=torch.int64)
    inc_mask = torch.zeros((N, K), dtype=torch.bool)
    src[0] = 0
    dst[0] = 1
    path[0, 0] = 0
    path[0, -1] = 1
    inc_idx[1, 0] = 0
    inc_mask[1, 0] = True

    fp = torch.zeros((D_FP,), dtype=torch.float32)
    md = torch.zeros((D_MD,), dtype=torch.float32)

    traced = torch.jit.trace(
        model,
        (begin_end, edge, vavn, mask_nodes, src, dst, path, vp, sl, mask_edges, inc_idx, inc_mask, fp, md),
        strict=False,
    )
    traced.save(TS_OUT)
    print("Saved TorchScript (trace):", TS_OUT)


if __name__ == "__main__":
    main()
