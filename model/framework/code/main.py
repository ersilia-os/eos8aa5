import os
import sys
import multiprocessing as mp
import numpy as np
import torch
from rdkit import Chem, DataStructs
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

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

# memory budget per worker on top of the shared model weights (graph, descriptors, activations)
MEM_PER_WORKER = 600 * 1024 ** 2
MAX_WORKERS = 16
# molecules per pool; workers are recreated after each batch so their memory cannot keep growing
BATCH_SIZE = 1000

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


def _available_memory() -> int:
    # container limit (cgroup v2, then v1) minus current usage, else MemAvailable of the host/VM
    for limit_file, usage_file in [
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ]:
        try:
            limit = open(limit_file).read().strip()
            usage = int(open(usage_file).read().strip())
            if limit != "max" and int(limit) < 2 ** 60:
                return int(limit) - usage
        except (OSError, ValueError):
            pass
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return MEM_PER_WORKER


def _n_workers() -> int:
    n_cpu = len(os.sched_getaffinity(0))
    n_mem = _available_memory() // MEM_PER_WORKER
    return int(max(1, min(n_cpu, n_mem, MAX_WORKERS)))


def _load_model(ts_path: str):
    # loaded once in the parent: forked workers share the 358 MB of weights copy-on-write
    global _G_MODEL

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    _G_MODEL = torch.jit.load(ts_path, map_location="cpu")
    _G_MODEL.eval()


def _init_worker():
    global _G_MDGEN

    torch.set_num_threads(1)
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

    outs = np.full((n, OUT_DIM), np.nan, dtype=np.float32)
    bad = 0
    failed = []

    _load_model(TS_PATH)
    n_workers = _n_workers()
    ctx = mp.get_context("fork")

    print(f"Parallel featurize+infer: n={n} workers={n_workers} batch_size={BATCH_SIZE}")
    start = 0
    while start < n:
        batch = list(enumerate(smiles_list[start:start + BATCH_SIZE], start=start))
        try:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_init_worker) as ex:
                results = list(ex.map(featurize_and_infer, batch))
        except BrokenProcessPool:
            # a worker was killed (usually OOM): retry the same batch with half the workers
            if n_workers == 1:
                raise
            n_workers = max(1, n_workers // 2)
            print(f"Worker died, retrying batch at {start} with workers={n_workers}")
            continue
        for idx, y in results:
            if y is None:
                bad += 1
                if len(failed) < 10:
                    failed.append(smiles_list[idx])
            else:
                outs[idx] = y
        start += BATCH_SIZE

    headers = [f"feat_{str(j).zfill(4)}" for j in range(outs.shape[1])]
    if output_file.endswith(".csv"):
        # failed rows are written as empty cells, not the string "nan"
        rows = [[None if np.isnan(v) else v for v in row] for row in outs]
        write_out(rows, headers, output_file, dtype=np.float32)
    else:
        write_out(outs, headers, output_file, dtype=np.float32)

    print(f"Done. inputs={n} outputs={outs.shape[0]} bad={bad}")
    if bad:
        print("Examples (up to 10):")
        for x in failed[:10]:
            print("  ", x)


if __name__ == "__main__":
    main()
