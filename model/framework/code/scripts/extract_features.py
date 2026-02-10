import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys

# Ensure src is importable
sys.path.append("..") 
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.collator import Collator_tune
from src.model.light import LiGhTPredictor as LiGhT
from src.model_config import config_dict

class SimpleListDataset(Dataset):
    """
    A simple dataset wrapper to mimic MoleculeDataset behavior 
    for the DataLoader and Collator.
    """
    def __init__(self, graphs, ecfp, md):
        self.graphs = graphs
        self.ecfp = ecfp
        self.md = md
        self.length = len(graphs)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Convert numpy arrays to Float Tensors (Models typically expect Float32)
        ecfp_tensor = torch.tensor(self.ecfp[idx], dtype=torch.float32)
        md_tensor = torch.tensor(self.md[idx], dtype=torch.float32)       
        dummy_label = torch.tensor([0.0], dtype=torch.float32)
               
        return (None, self.graphs[idx], ecfp_tensor, md_tensor, dummy_label)

def run_light_inference(config_name, model_path, graphs, ecfp_array, md_array):
    """
    Refactored to use DataLoader and Collator_tune to match exact processing logic.
    """
    config = config_dict[config_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size
        ).to(device)

    # 2. Load Weights
    map_location = torch.device('cpu') if str(device) == "cpu" else None
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
    model.eval()

    # 3. Setup Data Loader with Collator
    collator = Collator_tune(config['path_length'])
    dataset = SimpleListDataset(graphs, ecfp_array, md_array)
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, 
                        num_workers=0, drop_last=False, collate_fn=collator)

    # 4. Inference Loop
    fps_list = []
    print(f"Running inference on {len(dataset)} items...")
    
    with torch.no_grad():
        for batch_idx, batched_data in enumerate(loader):
            # Unpack exactly as original script
            (_, g, ecfp, md, labels) = batched_data
            mx_n = 0
            mx_e = 0
            # for g in graphs[:100]:
            #     mx_n = max(mx_n, g.num_nodes())
            #     mx_e = max(mx_e, g.num_edges())
            # print("max_nodes", mx_n, "max_edges", mx_e)
            
            ecfp = ecfp.to(device)
            md = md.to(device)
            g = g.to(device)
            
            fps = model.generate_fps(g, ecfp, md)
            fps_list.extend(fps.detach().cpu().numpy().tolist())
            
    return np.array(fps_list, dtype=np.float32)