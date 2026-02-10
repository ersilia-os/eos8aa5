import os
import sys
import numpy as np
import torch
from rdkit import Chem, DataStructs
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_config import config_dict
from src.data.featurizer import smiles_to_graph_tune
from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from ersilia_pack_utils.core import read_smiles, write_out

CONFIG_NAME = "base"
CFG = config_dict[CONFIG_NAME]

D_FP = 512
D_MD = 200
PATH_LEN = CFG["path_length"]
N_VIRTUAL_NODES = 2

OUT_DIM = CFG["d_g_feats"] * 3
N_WORKERS = 16#max(1, (os.cpu_count() or 4) // 2)

input_file = sys.argv[1]
output_file = sys.argv[2]

root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(root, "..", ".."))
TS_PATH = os.path.join(project_root, "checkpoints", "pretrained", CONFIG_NAME, "light.ts.pt")

device = torch.device("cpu")

_G_MODEL = None
_G_MDGEN = None


def build_incidence_fast(dst: torch.Tensor, N: int):
    E = int(dst.numel())
    if E == 0:
        inc_idx = torch.full((N, 1), -1, dtype=torch.int64)
        inc_mask = torch.zeros((N, 1), dtype=torch.bool)
        return inc_idx, inc_mask

    edge_ids = torch.arange(E, dtype=torch.int64)

    dst_sorted, perm = torch.sort(dst)
    e_sorted = edge_ids[perm]

    counts = torch.bincount(dst, minlength=N).to(torch.int64)
    Kmax = int(torch.max(counts).item())
    if Kmax < 1:
        Kmax = 1

    starts = torch.cumsum(counts, dim=0) - counts
    j = torch.arange(E, dtype=torch.int64)
    pos_in_group = j - starts[dst_sorted]

    inc_idx = torch.full((N, Kmax), -1, dtype=torch.int64)
    inc_mask = torch.zeros((N, Kmax), dtype=torch.bool)

    inc_idx[dst_sorted, pos_in_group] = e_sorted
    inc_mask[dst_sorted, pos_in_group] = True
    return inc_idx, inc_mask


def _init_worker(ts_path: str):
    global _G_MODEL, _G_MDGEN

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    _G_MODEL = torch.jit.load(ts_path, map_location="cpu")
    _G_MODEL.eval()

    _G_MDGEN = RDKit2DNormalized()


def _fp_from_smiles(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((D_FP,), dtype=np.float32)

    bv = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=D_FP)
    arr = np.zeros((D_FP,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)  
    return arr.astype(np.float32)


def _md_from_smiles(smiles: str) -> np.ndarray:
    md = _G_MDGEN.process(smiles)
    md = np.asarray(md[1:], dtype=np.float32)

    if md.shape[0] < D_MD:
        out = np.zeros((D_MD,), dtype=np.float32)
        out[: md.shape[0]] = md
        return out
    return md[:D_MD].astype(np.float32, copy=False)


def featurize_and_infer(idx_smiles):
    global _G_MODEL, _G_MDGEN
    idx, smiles = idx_smiles

    try:
        g = smiles_to_graph_tune(smiles, max_length=PATH_LEN, n_virtual_nodes=N_VIRTUAL_NODES)
        if g is None:
            return idx, None

        begin_end = g.ndata["begin_end"].to(dtype=torch.float32, device="cpu")
        edge = g.ndata["edge"].to(dtype=torch.float32, device="cpu")
        vavn = g.ndata["vavn"].to(dtype=torch.int64, device="cpu")

        src, dst = g.edges()
        src = src.to(dtype=torch.int64, device="cpu")
        dst = dst.to(dtype=torch.int64, device="cpu")

        path = g.edata["path"].to(dtype=torch.int64, device="cpu")
        vp = g.edata["vp"].to(dtype=torch.bool, device="cpu")
        sl = g.edata["sl"].to(dtype=torch.bool, device="cpu")

        Nn = int(begin_end.shape[0])
        Ee = int(src.shape[0])

        mask_nodes = torch.ones((Nn,), dtype=torch.bool)
        mask_edges = torch.ones((Ee,), dtype=torch.bool)

        inc_idx, inc_mask = build_incidence_fast(dst, Nn)

        fp = torch.from_numpy(_fp_from_smiles(smiles)).to(dtype=torch.float32)
        md = torch.from_numpy(_md_from_smiles(smiles)).to(dtype=torch.float32)

        with torch.no_grad():
            y = _G_MODEL(
                begin_end, edge, vavn, mask_nodes,
                src, dst, path, vp, sl, mask_edges,
                inc_idx, inc_mask,
                fp, md
            )

        return idx, y.detach().cpu().numpy().astype(np.float32, copy=False)

    except Exception:
        return idx, None


def main():
    _, smiles_list = read_smiles(input_file)
    smiles_list = list(smiles_list)
    n = len(smiles_list)

    outs = np.zeros((n, OUT_DIM), dtype=np.float32)
    bad = 0
    failed = []

    print(f"Parallel featurize+infer: n={n} workers={N_WORKERS}")
    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_init_worker,
        initargs=(TS_PATH,),
    ) as ex:
        futures = [ex.submit(featurize_and_infer, (i, s)) for i, s in enumerate(smiles_list)]
        for fut in as_completed(futures):
            idx, y = fut.result()
            if y is None:
                bad += 1
                if len(failed) < 10:
                    failed.append(smiles_list[idx])
            else:
                outs[idx] = y

    headers = [f"dim_{str(j).zfill(4)}" for j in range(outs.shape[1])]
    write_out(outs, headers, output_file, dtype=np.float32)

    print(f"Done. inputs={n} outputs={outs.shape[0]} bad={bad}")
    if bad:
        print("Examples (up to 10):")
        for x in failed[:10]:
            print("  ", x)


if __name__ == "__main__":
    main()
