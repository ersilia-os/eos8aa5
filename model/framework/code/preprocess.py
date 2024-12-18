# imports
import os
import csv
import sys

# parse arguments
input_file = sys.argv[1]
output_dir = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

# create output file
output_file = os.path.join(output_dir, "data", "data.csv")
os.mkdir(os.path.join(output_dir, "data"))
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])
    for smiles in smiles_list:
        writer.writerow([smiles])
