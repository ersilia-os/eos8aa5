import os
import csv
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch 

# --- Model Configuration Imports ---
from src.model_config import config_dict 

# Utility Imports
from ersilia_pack_utils.core import write_out, read_smiles

# Import the refactored functions
from scripts.preprocess_downstream_dataset import extract_features_from_smiles
from scripts.extract_features import run_light_inference 

# --- Configuration and Path Setup ---

if len(sys.argv) < 3:
    print("Usage: python main.py <input_file_path> <output_file_path>")
    sys.exit(1)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

CONFIG_NAME = 'base' 
CONFIG = config_dict[CONFIG_NAME] 
CHECKPOINT_PATH = os.path.join(project_root, 'checkpoints', 'pretrained', CONFIG_NAME, f'{CONFIG_NAME}.pth')

input_file = sys.argv[1]
output_file = sys.argv[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------

def my_model(smiles_list):
    """
    1. Extracts all necessary features using the imported featurization function.
    2. Runs LiGhT model inference using the imported inference function.
    3. Maps the features back to the original list length.
    """
    
    # Set seed for reproducibility
    # torch.manual_seed(22)
    np.random.seed(22)
    
    # --- 1. Feature Extraction ---
    print("Starting feature extraction...")
    feature_data = extract_features_from_smiles(smiles_list, 
                                                n_jobs=16, 
                                                path_length=CONFIG['path_length'])
    
    graphs = feature_data['graphs']
    # CHANGE IS HERE: Use 'ecfp' key
    ecfp_arr = feature_data['ecfp']
    md_arr = feature_data['descriptors']
    valid_indices = feature_data['valid_indices'] 
    
    # Handle case with no valid graphs
    if not graphs:
        print("Warning: No valid graphs were generated. Returning zero features.")
        return np.zeros((len(smiles_list), 512), dtype=np.float32)

    # --- 2. Model Loading and Inference ---
    print('Running LiGhT model inference...')

    X_valid = run_light_inference(
        config_name=CONFIG_NAME,
        model_path=CHECKPOINT_PATH,
        graphs=graphs,
        ecfp_array=ecfp_arr,
        md_array=md_arr
    )
    
    # --- 3. Map Features Back ---
    FEATURE_DIM = X_valid.shape[1]
    X_final = np.zeros((len(smiles_list), FEATURE_DIM), dtype=np.float32)
    
    X_final[valid_indices] = X_valid
    
    print(f"Successfully extracted and mapped {len(X_valid)} features of dimension {FEATURE_DIM} back to {len(smiles_list)} inputs.")
    
    return X_final

# --- Execution Flow ---

# 1. Read input SMILES and headers
cols, smiles_list = read_smiles(input_file)

# 2. Run the model
outputs = my_model(smiles_list)

# 3. Validation
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len, f"Input length ({input_len}) must equal output length ({output_len})"

# 4. Define headers
headers = ["dim_{0}".format(str(i).zfill(4)) for i in range(outputs.shape[1])]

# 5. Write the final output
write_out(outputs, headers, output_file, dtype='float32')