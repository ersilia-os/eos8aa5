import numpy as np
from multiprocessing import Pool
from dgllife.utils.io import pmap
from rdkit import Chem
# Note: We use Chem.RDKFingerprint to match the original script exactly.
# even though the inference script calls the variable 'ecfp'.
from src.data.featurizer import smiles_to_graph_tune
from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

def extract_features_from_smiles(smiles_list, n_jobs=16, path_length=5):
    """
    Refactored to match original logic exactly:
    - Uses RDKit Topological Fingerprints (not Morgan/ECFP).
    - Returns valid indices for mapping.
    """
    
    # 1. Graph Construction
    print('Constructing graphs')
    graphs = pmap(smiles_to_graph_tune,
                  smiles_list,
                  max_length=path_length,
                  n_virtual_nodes=2,
                  n_jobs=n_jobs)
    
    # Filter valid graphs
    valid_graphs = [g for g in graphs if g is not None]
    valid_indices = [i for i, g in enumerate(graphs) if g is not None]
    
    # 2. Fingerprint Extraction (RDKit Topological)
    # The original script used Chem.RDKFingerprint, so we must use it too.
    print('Extracting fingerprints (RDKit Topological)')
    FP_list = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # MATCHING ORIGINAL EXACTLY: minPath=1, maxPath=7, fpSize=512
            fp = list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512))
        else:
            fp = [0] * 512
        FP_list.append(fp)
        
    # Although the inference script calls this 'ecfp', it contains RDKit fingerprints.
    # We convert to float32 as Torch expects floats.
    FP_arr = np.array(FP_list, dtype=np.float32)
    
    # 3. Molecular Descriptors Extraction
    print('Extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(n_jobs).imap(generator.process, list(smiles_list))
    
    md_arr = np.array(list(features_map))
    # Remove first column (SMILES string)
    molecular_descriptors = md_arr[:, 1:].astype(np.float32)
    
    return {
        'graphs': valid_graphs,
        'ecfp': FP_arr[valid_indices], # Passing RDKit FPs as 'ecfp'
        'descriptors': molecular_descriptors[valid_indices],
        'valid_indices': valid_indices
    }